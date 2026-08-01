

import os
import argparse
import pandas as pd

from forward_model import predict_tg
from paths import DEFAULT_FORWARD_MODEL, DEFAULT_RESULTS_DIR

DEFAULT_INPUT_PATH = str(DEFAULT_RESULTS_DIR / "generated.csv")
DEFAULT_MODEL_PATH = str(DEFAULT_FORWARD_MODEL)
SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "SMILES-Repeating unit"]
TARGET_COLUMN_CANDIDATES = ["tg_target", "Tg( deg C)", "Tg(°C)", "Tg"]


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    ext = ext or ".csv"
    return f"{base}_with_pred_tg{ext}"


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main():
    parser = argparse.ArgumentParser(description="Predict Tg values for a generated-result CSV")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Input generated-result CSV path; default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path; default creates *_with_pred_tg.csv next to the input file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Forward Tg prediction model path; default: {DEFAULT_MODEL_PATH}",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output or default_output_path(input_path))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    smiles_col = pick_first_existing_column(df, SMILES_COLUMN_CANDIDATES)
    if smiles_col is None:
        raise ValueError(
            "Input CSV is missing a recognized SMILES column. Expected one of: "
            + ", ".join(SMILES_COLUMN_CANDIDATES)
        )

    smiles_list = df[smiles_col].fillna("").astype(str).tolist()
    preds = predict_tg(smiles_list, model_path=args.model)

    df["pred_tg"] = preds
    target_col = pick_first_existing_column(df, TARGET_COLUMN_CANDIDATES)
    if target_col is not None:
        df["tg_error"] = df["pred_tg"] - df[target_col]
        df["tg_abs_error"] = (df["pred_tg"] - df[target_col]).abs()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    pred_count = int(df["pred_tg"].notna().sum())
    print(f"Input file: {input_path}")
    print(f"SMILES column: {smiles_col}")
    print(f"Output file: {output_path}")
    print(f"Successful predictions: {pred_count}/{len(df)}")
    if pred_count > 0:
        print(
            "Predicted Tg statistics: "
            f"mean={df['pred_tg'].mean():.1f}  "
            f"min={df['pred_tg'].min():.1f}  "
            f"max={df['pred_tg'].max():.1f}"
        )
        if "tg_abs_error" in df.columns:
            print(f"Mean absolute target Tg error: {df['tg_abs_error'].mean():.1f} deg C")


if __name__ == "__main__":
    main()
