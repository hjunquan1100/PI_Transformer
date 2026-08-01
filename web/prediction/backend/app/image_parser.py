"""Structure image to SMILES via DECIMER-V2 (main model only).

Avoids `import DECIMER` which eagerly downloads/loads the HandDrawn model (~285MB).
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Callable

_predict_smiles: Callable[..., str] | None = None

_DECIMER_MAIN_URL = "https://zenodo.org/record/8300489/files/models.zip"
_DECIMER_HOME = Path.home() / ".data" / "DECIMER-V2"
_DECIMER_MODEL_PB = _DECIMER_HOME / "DECIMER_model" / "saved_model.pb"


class ImageParseError(ValueError):
    """DECIMER failed to parse the structure image."""


def _decimer_site_dir() -> Path:
    for p in sys.path:
        cand = Path(p) / "DECIMER"
        if (cand / "utils.py").is_file():
            return cand
    raise ImportError("DECIMER package not found. Run: pip install decimer")


def _load_decimer_module(name: str, filename: str):
    path = _decimer_site_dir() / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decimer_main_model_ready() -> bool:
    return _DECIMER_MODEL_PB.is_file() and _DECIMER_MODEL_PB.stat().st_size > 1_000_000


def _decimer_import_error() -> str | None:
    try:
        _decimer_site_dir()
    except ImportError as exc:
        return str(exc)
    if not _decimer_main_model_ready():
        return (
            "DECIMER main model is not downloaded. Run:\n"
            "  python -c \"import pystow; from pathlib import Path; "
            "import importlib.util; "
            "u=importlib.util.spec_from_file_location('u', str(Path(pystow.__file__).parent.parent/'DECIMER/utils.py')); "
            "m=importlib.util.module_from_spec(u); u.loader.exec_module(m); "
            f"m.ensure_models('{_DECIMER_HOME}', {{'DECIMER': '{_DECIMER_MAIN_URL}'}})\""
        )
    return None


def decimer_available() -> bool:
    """Lightweight check: package present + main model on disk (no TF load)."""
    return _decimer_import_error() is None


def _load_predict_smiles() -> Callable[..., str]:
    global _predict_smiles
    if _predict_smiles is not None:
        return _predict_smiles

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import tensorflow as tf
    import pystow

    utils = _load_decimer_module("decimer_utils", "utils.py")
    pre_process = _load_decimer_module("decimer_pre_process", "pre_process.py")

    default_path = pystow.join("DECIMER-V2")
    model_paths = utils.ensure_models(
        default_path=default_path,
        model_urls={"DECIMER": _DECIMER_MAIN_URL},
    )
    model_dir = model_paths["DECIMER"]
    tokenizer_path = os.path.join(model_dir, "assets", "tokenizer_SMILES.pkl")

    def load_tokenizer(tokenizer_path: str) -> object:
        try:
            with open(tokenizer_path, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            if "keras.preprocessing" in str(e):

                class _Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("keras."):
                            module = module.replace("keras.", "tensorflow.keras.")
                        return super().find_class(module, name)

                with open(tokenizer_path, "rb") as f:
                    return _Unpickler(f).load()
            raise

    tokenizer = load_tokenizer(tokenizer_path)
    decimer_v2 = tf.saved_model.load(model_dir)

    def _detokenize(predicted_array) -> str:
        outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
        return (
            "".join(str(elem) for elem in outputs)
            .replace("<start>", "")
            .replace("<end>", "")
        )

    def predict_smiles(image_input: str, hand_drawn: bool = False) -> str:
        if hand_drawn:
            raise ImageParseError("Only standard structure images are supported by this package.")
        chemical_structure = pre_process.decode_image(image_input)
        predicted_tokens, _ = decimer_v2(tf.constant(chemical_structure))
        return utils.decoder(_detokenize(predicted_tokens))

    _predict_smiles = predict_smiles
    return _predict_smiles


def image_bytes_to_smiles(image_bytes: bytes, suffix: str = ".png") -> str:
    err = _decimer_import_error()
    if err:
        raise ImageParseError(f"{err}")

    predict_smiles = _load_predict_smiles()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        smiles = predict_smiles(tmp_path, hand_drawn=False)
    except Exception as exc:
        raise ImageParseError(f"Structure image recognition failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not str(smiles).strip():
        raise ImageParseError("DECIMER did not recognize a valid SMILES from the image.")

    return str(smiles).strip()
