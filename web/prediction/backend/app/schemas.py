"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SmilesPredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="Polymer repeat-unit SMILES")


class SmilesPredictResponse(BaseModel):
    smiles: str
    tg_celsius: float
    unit: str = "C"


class ImagePredictResponse(BaseModel):
    smiles: str
    tg_celsius: float
    unit: str = "C"
    parsed_from: str = "image"
    recognized_smiles: str | None = None
    matched_reference: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    decimer_available: bool
    device: str
    inverse_model_loaded: bool = False
    inverse_ckpt: str | None = None


class InverseGenerateRequest(BaseModel):
    tg_target_c: float = Field(..., description="Target glass transition temperature (deg C)")


class GeneratedMolecule(BaseModel):
    id: str
    smiles: str
    tg_target_c: float
    pred_tg_c: float
    tg_error_c: float
    rank: int
    valid: bool = True


class InverseGenerateResponse(BaseModel):
    tg_target_c: float
    n_generated: int
    valid_count: int
    passed_count: int
    recommended: list[GeneratedMolecule]
    others: list[GeneratedMolecule]
