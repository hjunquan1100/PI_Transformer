# Terminal Demo and Web Launch

## Environment

```bash
cd UniPolymer
bash scripts/setup_env.sh
```

This creates or reuses the `pi` conda environment, installs Python dependencies, and downloads the DECIMER main model for image recognition.

## Terminal Demo

```bash
conda activate pi
bash scripts/demo.sh --pause
```

Equivalent command:

```bash
bash scripts/run_demo.sh --tg_target 300 --n_samples 15 --max_abs_error 25 --top_k 5 --pause
```

Arguments:

- `--tg_target`: target Tg in deg C, default 300
- `--n_samples`: number of generated candidates, default 20
- `--max_abs_error`: maximum absolute Tg error for final display, default 25 deg C
- `--top_k`: maximum number of final structures, default 5
- `--pause`: wait for Enter between steps
- `--out_dir`: custom output directory

Pipeline:

1. Generate target-conditioned structures and PNG images.
2. Parse PNG structures through DECIMER and predict Tg with TransPolymer.
3. Apply L1/L2/L3 chemical filters and write final ranked structures.

Outputs are written to `results/demo_YYYYMMDD_HHMMSS/`.

## Web App

```bash
cd UniPolymer
bash web/prediction/scripts/start-web.sh
```

Open `http://localhost:5173`.

- `/`: Tg prediction
- `/inverse`: inverse design

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```
