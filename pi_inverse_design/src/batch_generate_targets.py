"""
batch_generate_targets.py

Generate candidate structures for multiple target Tg values and summarize pass rates.

Example:
    python src/batch_generate_targets.py \
        --targets 200 300 400 \
        --n_samples 1000 \
        --out_dir results/batch_generation
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from generate import (
    decode_tokens,
    load_model,
    make_result_id,
    save_results_csv,
    selfies_to_valid_smiles,
    tg_to_tensors,
)
from paths import DEFAULT_GENERATOR_CKPT
from validator import filter_generated_results, print_report


DEFAULT_CKPT = str(DEFAULT_GENERATOR_CKPT)
DEFAULT_PLOT_NAME = "pass_rate_plot.png"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_dir(base_dir: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"batch_targets_{timestamp}")
    ensure_dir(run_dir)
    return run_dir


def summarize_target(
    tg_target: float,
    n_samples: int,
    output: dict,
) -> dict:
    valid_count = int(output["valid_count"])
    passed_count = int(len(output["passed_results"]))
    rdkit_valid_rate = valid_count / max(n_samples, 1)
    pass_rate_all = passed_count / max(n_samples, 1)
    pass_rate_valid = passed_count / max(valid_count, 1)
    return {
        "tg_target": tg_target,
        "n_samples": n_samples,
        "rdkit_valid_count": valid_count,
        "rdkit_valid_rate": round(rdkit_valid_rate, 4),
        "pi_pass_count": passed_count,
        "pi_pass_rate_all": round(pass_rate_all, 4),
        "pi_pass_rate_valid": round(pass_rate_valid, 4),
    }


def save_summary_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "tg_target",
        "n_samples",
        "rdkit_valid_count",
        "rdkit_valid_rate",
        "pi_pass_count",
        "pi_pass_rate_all",
        "pi_pass_rate_valid",
        "all_csv",
        "passed_csv",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_summary_csv(path: str) -> list[dict]:
    rows = []
    numeric_fields = {
        "tg_target",
        "n_samples",
        "rdkit_valid_count",
        "rdkit_valid_rate",
        "pi_pass_count",
        "pi_pass_rate_all",
        "pi_pass_rate_valid",
    }
    int_fields = {
        "n_samples",
        "rdkit_valid_count",
        "pi_pass_count",
    }

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in numeric_fields and value not in (None, ""):
                    number = float(value)
                    parsed[key] = int(number) if key in int_fields else number
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def configure_plot_style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidate_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "AR PL SungtiL GB",
        "AR PL KaitiM GB",
        "AR PL Mingti2L Big5",
        "AR PL KaitiM Big5",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [name for name in candidate_fonts if name in available_fonts]

    plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, selected_fonts


def plot_pass_rate_summary(rows: list[dict], output_path: str) -> list[str]:
    if not rows:
        raise ValueError("Summary is empty; cannot create pass-rate plot")

    plt, selected_fonts = configure_plot_style()

    tg_targets = [row["tg_target"] for row in rows]
    rdkit_valid_rates = [row["rdkit_valid_rate"] * 100 for row in rows]
    pi_pass_rates = [row["pi_pass_rate_all"] * 100 for row in rows]
    x_positions = list(range(len(rows)))
    bar_width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_bars = ax.bar(
        [x - bar_width / 2 for x in x_positions],
        rdkit_valid_rates,
        width=bar_width,
        color="#3b789b",
        edgecolor="#333333",
        label="RDKit valid",
    )
    pass_bars = ax.bar(
        [x + bar_width / 2 for x in x_positions],
        pi_pass_rates,
        width=bar_width,
        color="#c95f38",
        edgecolor="#333333",
        label="PI passed",
    )

    ax.set_title("Candidate Pass Rates by Target Tg", fontsize=16, pad=18)
    ax.set_xlabel("Target Tg (deg C)", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(round(tg))}" for tg in tg_targets], fontsize=11)
    ax.set_ylim(0, max(max(rdkit_valid_rates), max(pi_pass_rates)) + 12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)

    for bars in (valid_bars, pass_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#333333",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return selected_fonts


@torch.no_grad()
def sample_top_p_parallel(
    model,
    tg_norm_t: torch.Tensor,
    tg_bin_t: torch.Tensor,
    vocab: dict,
    inv_vocab: dict,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_len: int,
    device: str,
) -> list[tuple[str | None, str]]:
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]

    tg_norm_batch = tg_norm_t.expand(batch_size)
    tg_bin_batch = tg_bin_t.expand(batch_size)
    tokens = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(tg_norm_batch, tg_bin_batch, tokens)[:, -1, :]
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumsum - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
        next_tok = sorted_idx.gather(1, sampled_pos).squeeze(1)
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)

        tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
        finished = finished | next_tok.eq(eos_id)
        if finished.all():
            break

    results = []
    token_rows = tokens.detach().cpu().tolist()
    for row in token_rows:
        selfies_str = decode_tokens(row, inv_vocab, bos_id, eos_id)
        smiles = selfies_to_valid_smiles(selfies_str)
        results.append((smiles, selfies_str))
    return results


def generate_batch_parallel(
    model,
    tg_target: float,
    vocab: dict,
    inv_vocab: dict,
    stats: dict,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_len: int,
    device: str,
    batch_size: int,
    verbose: bool,
) -> dict:
    tg_norm_t, tg_bin_t = tg_to_tensors(tg_target, stats, device)
    results = []
    valid_count = 0
    generated = 0

    if verbose:
        print(f"\nGenerating {n_samples} candidates for target Tg = {tg_target} deg C")

    while generated < n_samples:
        current_batch = min(batch_size, n_samples - generated)
        batch_results = sample_top_p_parallel(
            model=model,
            tg_norm_t=tg_norm_t,
            tg_bin_t=tg_bin_t,
            vocab=vocab,
            inv_vocab=inv_vocab,
            batch_size=current_batch,
            temperature=temperature,
            top_p=top_p,
            max_len=max_len,
            device=device,
        )

        for smiles, selfies_str in batch_results:
            is_valid = smiles is not None
            if is_valid:
                valid_count += 1
            results.append(
                {
                    "id": make_result_id(len(results) + 1),
                    "smiles": smiles if is_valid else "",
                    "selfies": selfies_str,
                    "tg_target": tg_target,
                    "valid": is_valid,
                }
            )

        generated += current_batch
        if verbose:
            print(
                f"  [{generated}/{n_samples}] "
                f"RDKit valid: {valid_count / max(generated, 1) * 100:.1f}%"
            )

    passed_results, all_vr = filter_generated_results(results, strict=True)
    if verbose:
        print_report(all_vr)
        print(f"PI-plausible candidates: {len(passed_results)} / {valid_count} RDKit-valid structures")

    return {
        "all_results": results,
        "passed_results": passed_results,
        "valid_count": valid_count,
        "pass_rate": round(len(passed_results) / max(valid_count, 1), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate candidates for multiple target Tg values")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Generator checkpoint path")
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Existing summary.csv path; create only the pass-rate plot",
    )
    parser.add_argument(
        "--plot_out",
        default=None,
        help="Pass-rate plot output path; default is the run directory",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[200.0, 300.0, 400.0],
        help="Target Tg list, for example: --targets 200 300 400",
    )
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples to generate per target")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--max_len", type=int, default=300, help="Maximum generated sequence length")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Parallel sampling batch size",
    )
    parser.add_argument(
        "--out_dir",
        default="results/batch_generation",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device; default selects cuda if available, otherwise cpu",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-target progress output",
    )
    args = parser.parse_args()

    if args.summary_csv:
        plot_output_path = args.plot_out or os.path.join(
            os.path.dirname(args.summary_csv),
            DEFAULT_PLOT_NAME,
        )
        summary_rows = load_summary_csv(args.summary_csv)
        selected_fonts = plot_pass_rate_summary(summary_rows, plot_output_path)
        font_desc = ", ".join(selected_fonts) if selected_fonts else "DejaVu Sans"
        print(f"Pass-rate plot saved: {plot_output_path}")
        print(f"Plot font: {font_desc}")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(args.out_dir)
    model, vocab, inv_vocab, stats = load_model(args.ckpt, device=device)

    summary_rows = []

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Target Tg values: {args.targets}")
    print(f"Samples per target: {args.n_samples}")

    for tg_target in args.targets:
        print(f"\n{'=' * 60}")
        print(f"Generating Tg={tg_target:.0f} deg C")

        output = generate_batch_parallel(
            model=model,
            tg_target=tg_target,
            vocab=vocab,
            inv_vocab=inv_vocab,
            stats=stats,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            verbose=not args.quiet,
        )

        target_dir = os.path.join(run_dir, f"tg_{int(round(tg_target))}")
        ensure_dir(target_dir)
        all_csv_path = os.path.join(target_dir, "generated_all.csv")
        passed_csv_path = os.path.join(target_dir, "generated_passed.csv")
        save_results_csv(all_csv_path, output["all_results"])
        save_results_csv(passed_csv_path, output["passed_results"])

        target_summary = summarize_target(
            tg_target=tg_target,
            n_samples=args.n_samples,
            output=output,
        )
        target_summary["all_csv"] = all_csv_path
        target_summary["passed_csv"] = passed_csv_path
        with open(
            os.path.join(target_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(target_summary, f, indent=2, ensure_ascii=False)

        summary_rows.append(target_summary)
        print(
            "Complete: "
            f"RDKit valid {target_summary['rdkit_valid_count']}/{args.n_samples} "
            f"({target_summary['rdkit_valid_rate'] * 100:.1f}%), "
            f"PI passed {target_summary['pi_pass_count']}/{args.n_samples} "
            f"({target_summary['pi_pass_rate_all'] * 100:.1f}%), "
            f"PI pass rate among valid structures {target_summary['pi_pass_rate_valid'] * 100:.1f}%"
        )

    total_samples = len(summary_rows) * args.n_samples
    total_valid = sum(row["rdkit_valid_count"] for row in summary_rows)
    total_passed = sum(row["pi_pass_count"] for row in summary_rows)
    overall = {
        "targets": args.targets,
        "n_samples_per_target": args.n_samples,
        "total_samples": total_samples,
        "total_rdkit_valid_count": total_valid,
        "total_rdkit_valid_rate": round(total_valid / max(total_samples, 1), 4),
        "total_pi_pass_count": total_passed,
        "total_pi_pass_rate_all": round(total_passed / max(total_samples, 1), 4),
        "total_pi_pass_rate_valid": round(total_passed / max(total_valid, 1), 4),
    }

    summary_csv_path = os.path.join(run_dir, "summary.csv")
    summary_json_path = os.path.join(run_dir, "summary.json")
    save_summary_csv(summary_csv_path, summary_rows)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_target": summary_rows,
                "overall": overall,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 60}")
    print("Complete")
    print(
        f" RDKit valid: {total_valid}/{total_samples} "
        f"({overall['total_rdkit_valid_rate'] * 100:.1f}%)"
    )
    print(
        f" PI passed: {total_passed}/{total_samples} "
        f"({overall['total_pi_pass_rate_all'] * 100:.1f}%)"
    )
    print(
        "PI pass rate among valid structures: "
        f"{overall['total_pi_pass_rate_valid'] * 100:.1f}%"
    )

    plot_output_path = args.plot_out or os.path.join(run_dir, DEFAULT_PLOT_NAME)
    selected_fonts = plot_pass_rate_summary(summary_rows, plot_output_path)
    font_desc = ", ".join(selected_fonts) if selected_fonts else "DejaVu Sans"

    print(f"Pass-rate plot saved: {plot_output_path}")
    print(f"Plot font: {font_desc}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
