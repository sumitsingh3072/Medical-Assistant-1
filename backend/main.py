# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
from contextlib import asynccontextmanager
import os
from services.xray_service import process_xray, init_xray_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run on startup
    init_xray_model()
    yield
    # Run on shutdown (if needed)
    # cleanup() or release resources here
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict/xray/")
async def predict_xray(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save the uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        predictions = process_xray(temp_path, device="cpu")
        os.remove(temp_path)  # Clean up
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
