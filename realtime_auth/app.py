from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from realtime_auth.runtime import AuthConfig, BiometricAuthenticator


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = ROOT / "runtime_store"
STATIC_ROOT = Path(__file__).resolve().parent / "static"

config = AuthConfig(
    checkpoint_path=ROOT / "checkpoints" / "seed_1" / "model_best.pt",
    registry_path=RUNTIME_ROOT / "registry.json",
    embeddings_dir=RUNTIME_ROOT / "embeddings",
)
authenticator = BiometricAuthenticator(config=config)

app = FastAPI(title="DOMCS-EEG Real-Time Authentication", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": config.device,
        "checkpoint": str(config.checkpoint_path),
        "threshold": config.threshold,
        "min_windows": config.min_windows,
    }


@app.get("/api/users")
def list_users():
    return {"users": authenticator.store.list_users()}


async def _read_windows(file: UploadFile):
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded EEG file is empty")
    try:
        return authenticator.parse_uploaded_array(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _read_bytes(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return payload


async def _read_bytes_list(files: list[UploadFile]) -> list[bytes]:
    payloads = []
    for file in files:
        payloads.append(await _read_bytes(file))
    if not payloads:
        raise HTTPException(status_code=400, detail="No files uploaded")
    return payloads


@app.post("/api/enroll")
async def enroll(user_id: str = Form(...), eeg_file: UploadFile = File(...)):
    try:
        windows = await _read_windows(eeg_file)
        return authenticator.enroll(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/verify")
async def verify(user_id: str = Form(...), eeg_file: UploadFile = File(...)):
    try:
        windows = await _read_windows(eeg_file)
        return authenticator.verify(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/master/inspect")
async def inspect_master(master_npz: UploadFile = File(...)):
    try:
        payload = await _read_bytes(master_npz)
        return authenticator.inspect_master_npz(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/master/enroll")
async def enroll_from_master(
    user_id: str = Form(...),
    subject_id: str = Form(...),
    runs: str = Form(...),
    master_npz: UploadFile = File(...),
):
    try:
        payload = await _read_bytes(master_npz)
        windows = authenticator.parse_master_npz(payload, subject_id=subject_id, run_list=runs)
        return authenticator.enroll(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/master/verify")
async def verify_from_master(
    user_id: str = Form(...),
    subject_id: str = Form(...),
    runs: str = Form(...),
    master_npz: UploadFile = File(...),
):
    try:
        payload = await _read_bytes(master_npz)
        windows = authenticator.parse_master_npz(payload, subject_id=subject_id, run_list=runs)
        return authenticator.verify(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/edf/enroll")
async def enroll_from_edf(user_id: str = Form(...), eeg_files: list[UploadFile] = File(...)):
    try:
        payloads = await _read_bytes_list(eeg_files)
        windows = authenticator.parse_raw_edf(payloads)
        return authenticator.enroll(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/edf/verify")
async def verify_from_edf(user_id: str = Form(...), eeg_files: list[UploadFile] = File(...)):
    try:
        payloads = await _read_bytes_list(eeg_files)
        windows = authenticator.parse_raw_edf(payloads)
        return authenticator.verify(user_id=user_id.strip(), windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/identify")
async def identify(eeg_file: UploadFile = File(...)):
    try:
        windows = await _read_windows(eeg_file)
        return authenticator.identify(windows=windows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/settings/threshold")
def update_threshold(threshold: float = Form(...)):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    authenticator.config.threshold = threshold
    return {"status": "updated", "threshold": threshold}


def main():
    uvicorn.run("realtime_auth.app:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
