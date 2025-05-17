from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from contextlib import asynccontextmanager
import os
from services.xray_service import process_xray, init_xray_model
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, Query
import httpx


# Global variable to store the latest prediction
latest_xray_results = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_xray_model()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS settings
origins = ["*"]  # allow all origins for simplicity; adjust as needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # allow all headers
)

@app.post("/predict/xray/")
async def predict_xray(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        predictions = process_xray(temp_path, device="cpu")
        os.remove(temp_path)
        global latest_xray_results
        latest_xray_results = {label: float(prob) for label, prob in predictions}
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_latest_results/")
async def get_latest_results():
    if not latest_xray_results:
        return {"message": "No prediction results available yet."}
    return latest_xray_results

# Mock database of doctors
class Doctor(BaseModel):
    name: str
    specialty: str
    location: str
    phone: str
    lat: float
    lng: float

mock_doctors = [
    {
        "name": "Dr. Anjali Sharma",
        "specialty": "Dermatologist",
        "location": "Bangalore",
        "phone": "9999000011",
        "lat": 12.9716,
        "lng": 77.5946,
    },
    {
        "name": "Dr. Ravi Patel",
        "specialty": "Cardiologist",
        "location": "Bangalore",
        "phone": "8888777766",
        "lat": 12.9720,
        "lng": 77.5950,
    },
    {
        "name": "Dr. Seema Reddy",
        "specialty": "Neurologist",
        "location": "Bangalore",
        "phone": "7777888899",
        "lat": 12.9705,
        "lng": 77.5930,
    },
    {
        "name": "Dr. Aditya Rao",
        "specialty": "Dermatologist",
        "location": "Bangalore",
        "phone": "9999888877",
        "lat": 12.9690,
        "lng": 77.5920,
    },
]

@app.get("/api/search-doctors", response_model=List[Doctor])
async def search_doctors(
    location: str = Query(...),
    specialty: Optional[str] = Query(None),
):
    filtered = []

    for doc in mock_doctors:
        if location.lower() in doc["location"].lower():
            if specialty is None or specialty.lower() in doc["specialty"].lower():
                filtered.append(doc)

    enriched_doctors = []

    async with httpx.AsyncClient() as client:
        for doc in filtered:
            res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"{doc['location']}, India", "format": "json", "limit": 1},
                headers={"User-Agent": "doctor-finder"},
            )
            data = res.json()
            if data:
                lat = float(data[0]["lat"])
                lng = float(data[0]["lon"])
            else:
                lat = 0.0
                lng = 0.0

            enriched_doctors.append({**doc, "lat": lat, "lng": lng})

    return enriched_doctors
