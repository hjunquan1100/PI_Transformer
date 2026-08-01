# PI Tg Prediction and Design Web App

This web app combines two workflows:

- Tg prediction: SMILES or structure image to predicted Tg.
- Inverse design: target Tg to generated PI candidate structures.

## Structure

```text
web/prediction/
├── backend/
├── frontend/
├── scripts/
└── test_assets/
```

## Requirements

- Python 3.10+
- A Python environment with `torch`, `rdkit`, `transformers`, `selfies`, and FastAPI dependencies
- Node.js 18+ for the frontend
- Packaged model artifacts under `pi_forward_prediction/` and `pi_inverse_design/`

## Backend

```bash
cd web/prediction/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper:

```bash
bash web/prediction/scripts/start-backend.sh
```

## Frontend

```bash
cd web/prediction/frontend
npm install
npm run dev
```

On shared filesystems, use the helper script. It installs frontend dependencies under `/tmp/pi-tg-fe-dev`.

```bash
bash web/prediction/scripts/start-frontend.sh
```

Open `http://localhost:5173`.

## Start Both Services

```bash
bash web/prediction/scripts/start-web.sh
```

Stop services:

```bash
bash web/prediction/scripts/start-web.sh stop
```

## API Endpoints

- `GET /api/health`
- `POST /api/predict/smiles`
- `POST /api/predict/image`
- `POST /api/inverse/generate`
- `GET /api/structure/svg`
- `GET /api/structure/png`

## Environment Variables

- `PI_FORWARD_ROOT`: forward-prediction package root
- `PI_TG_CONFIG`: forward model YAML config
- `PI_TG_CKPT`: forward Tg checkpoint
- `PI_INVERSE_ROOT`: inverse-design package root
- `PI_INVERSE_CKPT`: inverse generator checkpoint
- `PI_INVERSE_FWD`: inverse-design forward evaluator checkpoint

## DECIMER

Image-to-SMILES recognition requires DECIMER and its main model files.

```bash
bash web/prediction/scripts/setup_decimer_model.sh
```

SMILES prediction and inverse generation do not require DECIMER.

## Smoke Tests

Run after the backend is listening on port 8000:

```bash
bash web/prediction/scripts/test_api.sh
bash web/prediction/scripts/test_inverse_api.sh
```
