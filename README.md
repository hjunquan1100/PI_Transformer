# UniPolymer

UniPolymer packages three reproducible components for polyimide (PI) glass-transition-temperature work:

Forward prediction:

- Source transfer dataset: `pi_forward_prediction/data/PI_Tg_10066.csv`
- Transfer-training data: `pi_forward_prediction/data/PI_Tg_10066_with_desc_p1_m128.csv`
- Holdout data: `pi_forward_prediction/data/PI_Tg_10066_holdout80_p1_m128.csv`
- Config: `pi_forward_prediction/experiments/config/p26_runs/kfold_m128_head15.yaml`
- Contrastive backbone: `pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/`
- Best Tg model: `pi_forward_prediction/ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt`

Inverse design:

- Source dataset copy: `pi_inverse_design/data/raw/data/PI_Tg_10066.csv`
- Raw dataset converted for the inverse-design scripts: `pi_inverse_design/data/raw/data/data.xlsx`
- Processed SELFIES dataset: `pi_inverse_design/data/processed/`
- Generator checkpoint: `pi_inverse_design/checkpoints/reverse_20260318_232126/best_model_valloss_0.9011.pt`
- GBR forward evaluator: `pi_inverse_design/checkpoints/forward_20260318_232230/forward_model_best_r2_0.8535.pkl`

`PI_Tg_10066.csv` is the unified project dataset. It contains 10066 rows with
`smiles` and `tg` columns. The forward transfer-training table expands it to
145 columns: `smiles`, `value`, 15 scalar descriptors, and `mfp_0` through
`mfp_127`. The inverse-design raw xlsx uses the same source rows. The packaged
processed SELFIES data contains 2290 unique RDKit-valid structures that the
current SELFIES encoder can represent.

Note: the packaged inverse generator checkpoint stores its own training stats
with `tg_min=19.0` and `tg_max=460.0`. To reproduce a generator trained on the
full `PI_Tg_10066.csv` range, rerun preprocessing and training, then replace the
checkpoint under `pi_inverse_design/checkpoints/`.

The model files are large. Use Git LFS before `git add`:

```bash
git lfs install
git lfs track "*.pt" "*.bin" "*.safetensors" "*.pkl" "*.joblib" "*.xlsx" "*.png"
```

`.gitattributes` is already included. Without Git LFS, GitHub will reject files over 100 MB.

## Environment

Recommended:

```bash
cd UniPolymer
conda create -n unipolymer python=3.10 -y
conda activate unipolymer
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

For CUDA machines, install the PyTorch wheel matching your CUDA version first, then run `pip install -r requirements.txt`.

The helper `bash scripts/setup_env.sh` can create a conda environment named `pi` and install DECIMER assets for image recognition.

## Forward Prediction

Run from the forward model directory:

```bash
cd pi_forward_prediction
python Downstream.py --config experiments/config/p26_runs/kfold_m128_head15.yaml
```

Primary dataset files:

- `data/PI_Tg_10066.csv`: source transfer-training data (`smiles`, `tg`).
- `data/PI_Tg_10066_with_desc_p1_m128.csv`: full 10066-row transfer-training table with descriptors.
- `data/PI_Tg_10066_holdout80_p1_m128.csv`: deterministic 80-row holdout with the same feature schema.
- `data/vocab/vocab_sup_PE_I.csv`: additional tokenizer vocabulary.

The web backend uses the packaged checkpoint directly for single-SMILES prediction.

## Inverse Design

Run from the inverse design directory:

```bash
cd pi_inverse_design
python src/preprocess.py --data data/raw/data/data.xlsx --out_dir data/processed
python src/forward_model.py --data data/raw/data/data.xlsx
python src/train.py --processed_dir data/processed --epochs 200 --batch_size 64
python src/generate.py --tg_target 300 --n_samples 100 --out results/generated_300.csv
```

To use the packaged generator checkpoint directly:

```bash
cd pi_inverse_design
python src/generate.py --tg_target 300 --n_samples 20 --out results/generated_300.csv
```

## Web App

Backend and frontend can be started together:

```bash
cd UniPolymer
bash web/prediction/scripts/start-web.sh
```

Open `http://localhost:5173`.

- `/`: SMILES/image to Tg prediction.
- `/inverse`: target Tg to candidate PI structures.

API smoke tests after the backend is running:

```bash
bash web/prediction/scripts/test_api.sh
bash web/prediction/scripts/test_inverse_api.sh
```

Image-to-SMILES requires DECIMER and its model files. SMILES prediction and inverse generation do not require DECIMER.

## Terminal Demo

```bash
cd UniPolymer
bash scripts/demo.sh
```

The demo writes outputs under `results/demo_YYYYMMDD_HHMMSS/`.
