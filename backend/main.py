from fastapi import FastAPI, UploadFile, File, HTTPException ,Path
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
from typing import List, Tuple
from dotenv import load_dotenv
import io


# Load environment variables
load_dotenv()

# Import your ML model functions for each modality
from services.xray_service import process_xray, init_xray_model
# Uncomment when available:
# from services.ct_service import process_ct, init_ct_model
# from services.ultrasound_service import process_ultrasound, init_ultrasound_model
# from services.mri_service import process_mri, init_mri_model

# Initialize Google GenAI Client (multimodal)
# pip install google-genai
from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global: store latest predictions for frontend polling
latest_xray_results: dict = {}
latest_reports = {}  

# Startup: initialize all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_xray_model()
    # init_ct_model()
    # init_ultrasound_model()
    # init_mri_model()
    yield
    print("Shutting down models...")

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


PROMPT_TEMPLATES = {
    "xray": (
        """"
        You are a medical AI assistant. 
        Given a set of confidence scores for these diseases, your task is to:

        1. Idenify if the image is of a chest X-ray. If not, return "Not a chest X-ray" and do not proceed.
        2. Identify the disease with the highest confidence score.
        3. Generate a clear and concise diagnosis statement indicating which disease is most likely present based on the AI analysis.
        4. Mention the confidence score as a percentage.
        5. Include a disclaimer that this is a preliminary AI-based diagnosis and advise the user to consult a healthcare professional for confirmation.
        6. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        7. Report size should be always between 200 and 300 words.
        8. Use the following format for the output:


        Example Output:
        Disease Expected: Mass
        The AI model analyzed the chest X-ray image and determined that the most likely condition present is Mass, with a confidence score of 47.00%. 
        This suggests there may be an abnormal growth or lump in the lung area that requires further attention. Masses can range from benign 
        (non-cancerous) to malignant (cancerous), so additional medical evaluation such as a CT scan or biopsy may be recommended to determine its 
        nature. This result is an early indication provided by an AI system and should not replace professional medical advice or diagnosis. 
        Please consult a certified radiologist or doctor.

        """
        "Based on the image and the patient symptoms: {symptoms}, "
        "provide a structured radiology report including findings, impression, and recommendations."
    ),
    "ct": (
        "You are a radiology report assistant specialized in interpreting CT scans. "
        "Based on the image and the patient symptoms: {symptoms}, "
        "produce a detailed CT scan report including observations, differential diagnoses, and next steps."
    ),
    "ultrasound": (
        "You are a radiology report assistant specialized in interpreting ultrasounds. "
        "Based on the image and the patient symptoms: {symptoms}, "
        "generate a comprehensive ultrasound report covering findings, clinical significance, and recommendations."
    ),
    "mri": (
        "You are a radiology report assistant specialized in interpreting MRI scans. "
        "Based on the image and the patient symptoms: {symptoms}, "
        "create a detailed MRI report including key findings, interpretation, and suggested follow‑up."
    ),
}
# A generic fallback if you ever get an unexpected modality:
FALLBACK_TEMPLATE = (
    "You are a medical report assistant. Based on the image and patient symptoms: {symptoms}, "
    "generate a concise professional report including findings and recommendations."
)
# Utility: extract top-k symptom labels
def extract_top_symptoms(predictions: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return [label for label, _ in sorted_preds[:top_k]]

# Generate report using multimodal Gemini
def generate_medical_report(symptoms: List[str], image_bytes: bytes, modality: str, mime_type: str = "image/png") -> str:
    # Prepare prompt
    template = PROMPT_TEMPLATES.get(modality.lower(), FALLBACK_TEMPLATE)
    prompt = template.format(symptoms=", ".join(symptoms))
    # prompt = (
    #     f"Based on the provided image and the following symptoms: {', '.join(symptoms)}, "
    #     "generate a clear, concise, and professional medical report. "
    #     "Include possible diagnoses, recommended next steps, and any relevant notes."
    # )
    # Wrap image bytes in Part for multimodal input
    from google.genai.types import Part
    image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)

    # Generate content with image part and prompt
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=[image_part, prompt]
    )
    if not response or not hasattr(response, 'text') or response.text is None:
        raise HTTPException(status_code=500, detail="Empty response from Gemini API.")
    return response.text




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
@app.post("/generate-report/{modality}/")
async def generate_report(
    modality: str = Path(..., description="One of: xray, ct, ultrasound, mri"),
    file: UploadFile = File(...)
):
    modality = modality.lower()
    if modality not in ["xray", "ct", "ultrasound", "mri"]:
        raise HTTPException(status_code=400, detail="Invalid modality.")
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{modality}_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        # Inference dispatch
        if modality == "xray":
            raw_preds = process_xray(temp_path, device="cpu")
        # elif modality == "ct": raw_preds = process_ct(temp_path, device="cpu")
        # elif modality == "ultrasound": raw_preds = process_ultrasound(temp_path, device="cpu")
        # else: raw_preds = process_mri(temp_path, device="cpu")

        symptoms = extract_top_symptoms(raw_preds)
        # Read bytes
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        report = generate_medical_report(symptoms, img_bytes, modality)
        # Store the report in a global variable
        latest_reports[modality] = {
        "symptoms": symptoms,
        "report": report
        }
        return JSONResponse(content={"symptoms": symptoms, "report": report})
    except HTTPException:
        os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-latest-report/{modality}/")
async def get_latest_report(modality: str = Path(...)):
    modality = modality.lower()
    if modality not in latest_reports:
        raise HTTPException(status_code=404, detail="No report available for this modality.")
    return latest_reports[modality]

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
