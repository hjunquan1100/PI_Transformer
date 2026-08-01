"""FastAPI application for PI glass transition temperature prediction."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import (
    ALLOWED_IMAGE_TYPES,
    CORS_ORIGINS,
    MAX_IMAGE_BYTES,
    PI_INVERSE_CKPT,
    PI_TG_CKPT,
)
from app.descriptors import (
    SmilesDescriptorError,
    canonicalize_smiles,
    same_compound,
    smiles_from_png_bytes,
    smiles_to_png_bytes,
    smiles_to_svg,
)
from app.image_parser import ImageParseError, decimer_available, image_bytes_to_smiles
from app.inference import get_predictor
from app.inverse_service import InverseGenerateError, inverse_model_loaded, run_inverse_generate
from app.schemas import (
    HealthResponse,
    ImagePredictResponse,
    InverseGenerateRequest,
    InverseGenerateResponse,
    SmilesPredictRequest,
    SmilesPredictResponse,
)

logger = logging.getLogger(__name__)

_predictor_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor_loaded
    logger.info("Loading Tg predictor from %s", PI_TG_CKPT)
    get_predictor()
    _predictor_loaded = True
    logger.info("Tg predictor ready.")
    yield


app = FastAPI(
    title="PI Tg Prediction & Inverse Design API",
    description="Polyimide glass transition temperature prediction and inverse design",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = "unknown"
    try:
        predictor = get_predictor()
        device = str(predictor.device)
        model_ok = True
    except Exception:
        model_ok = False
    decimer_ok = decimer_available()
    inv_ok = inverse_model_loaded()
    status = "ok" if model_ok else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model_ok,
        decimer_available=decimer_ok,
        device=device,
        inverse_model_loaded=inv_ok,
        inverse_ckpt=str(PI_INVERSE_CKPT) if PI_INVERSE_CKPT.is_file() else None,
    )


@app.post("/api/inverse/generate", response_model=InverseGenerateResponse)
def inverse_generate(body: InverseGenerateRequest) -> InverseGenerateResponse:
    try:
        result = run_inverse_generate(body.tg_target_c)
    except InverseGenerateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("inverse_generate failed")
        raise HTTPException(status_code=500, detail=f"Inverse generation failed: {exc}") from exc
    return InverseGenerateResponse(**result)


@app.post("/api/predict/smiles", response_model=SmilesPredictResponse)
def predict_smiles(body: SmilesPredictRequest) -> SmilesPredictResponse:
    try:
        smiles = canonicalize_smiles(body.smiles.strip())
        tg = get_predictor().predict_tg(smiles)
    except SmilesDescriptorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_smiles failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return SmilesPredictResponse(smiles=smiles, tg_celsius=round(tg, 2))


@app.post("/api/predict/image", response_model=ImagePredictResponse)
async def predict_image(
    file: UploadFile = File(...),
    reference_smiles: str | None = Form(None),
) -> ImagePredictResponse:
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Upload PNG or JPEG.",
        )
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image size must not exceed 5 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = ".png"
    if file.filename and file.filename.lower().endswith((".jpg", ".jpeg")):
        suffix = ".jpg"

    try:
        # Prefer PNG metadata written by this system.
        embedded = smiles_from_png_bytes(data) if suffix == ".png" else None
        if embedded:
            recognized = embedded
            parsed_from = "image_metadata"
        else:
            recognized = canonicalize_smiles(image_bytes_to_smiles(data, suffix=suffix))
            parsed_from = "image"

        # If the recognized image matches the reference SMILES, use the reference for consistent Tg.
        matched = False
        predict_smi = recognized
        ref = (reference_smiles or "").strip()
        if ref:
            try:
                ref_canon = canonicalize_smiles(ref)
                if same_compound(recognized, ref_canon):
                    predict_smi = ref_canon
                    matched = True
                    parsed_from = "image_matched_reference"
            except SmilesDescriptorError:
                pass

        tg = get_predictor().predict_tg(predict_smi)
    except ImageParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SmilesDescriptorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_image failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return ImagePredictResponse(
        smiles=predict_smi,
        tg_celsius=round(tg, 2),
        parsed_from=parsed_from,
        recognized_smiles=recognized if recognized != predict_smi else None,
        matched_reference=matched,
    )


@app.get("/api/structure/svg")
def structure_svg(smiles: str = Query(..., min_length=1)) -> Response:
    try:
        svg = smiles_to_svg(smiles.strip())
    except SmilesDescriptorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/structure/png")
def structure_png(smiles: str = Query(..., min_length=1)) -> Response:
    """Return a structure PNG with embedded SMILES metadata."""
    try:
        png = smiles_to_png_bytes(smiles.strip())
    except SmilesDescriptorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")
