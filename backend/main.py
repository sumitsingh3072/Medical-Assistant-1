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
import re
import numpy as np
import nibabel as nib # type: ignore
from PIL import Image
from nibabel.loadsave import load # type: ignore
from geopy.geocoders import Nominatim



# Load environment variables
load_dotenv()

# Import your ML model functions for each modality
from services.xray_service import process_xray, init_xray_model
# Uncomment when available:
from services.ct_service import process_ct, init_ct_models
from services.ultrasound_service import process_ultrasound, init_ultrasound_model
from services.mri_service import process_mri, init_mri_models

# Initialize Google GenAI Client (multimodal)
# pip install google-genai
from google import genai
from google.genai.types import Part
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global: store latest predictions for frontend polling
latest_xray_results: dict = {}
latest_reports = {}  

# Startup: initialize all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_xray_model()
    init_ct_models()
    init_ultrasound_model()
    init_mri_models()
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
        Based on the image and the patient symptoms: {symptoms} your task is to:

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
    
    ),
    "ct": (
        '''
        You are a medical AI assistant specialized in interpreting 3D and 2D CT scan results. 
        Given a set of AI-generated confidence scores for tumor detection, your task is to:

        1. Identify whether a tumor or no tumor is more likely based on the highest confidence score.
        2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 92.00%).
        3. Explain what this result means for the patient in clear, simple language.
        4. Describe briefly how 3D CT scans assist in detecting tumors by providing detailed cross-sectional views of the body.
        5. Recommend possible next steps such as further imaging or biopsy for confirmation.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:

        """
        Output example 
        Condition Detected: Tumor
        The AI analysis of your 3D CT scan of the brain indicates a high probability of a tumor, with a confidence score of 92.00%. This suggests there may be an abnormal mass or growth
        present in the scanned region. 3D CT scans allow doctors to view detailed cross-sectional images of internal tissues, making it easier to identify potential issues like 
        tumors. While this result is a strong indicator, it is not a confirmed diagnosis. Further testing, such as an MRI or biopsy, may be required. 
        Disclaimer: This is an AI-generated summary. Please consult a certified doctor or radiologist for medical confirmation and advice.
        '''
        "Based on the image and the patient symptoms: {symptoms}, "
        "produce a detailed CT scan report including observations, differential diagnoses, and next steps."
    ),
    "ultrasound": (
        '''

        You are a medical assistant specialized in interpreting ultrasound scan results. 
        Based on the image and the patient symptoms: {symptoms}, your task is to:

        1. Identify the most likely condition based on the highest confidence score from the following categories: Normal, Cyst, Mass, Fluid, Other Anomaly.
        2. Clearly state the detected condition along with its confidence score as a percentage (e.g., 88.50%).
        3. Explain in simple and compassionate language what the result implies for the patient.
        4. Provide a brief explanation of how ultrasound helps in detecting such conditions using sound waves for real-time internal imaging.
        5. Suggest next medical steps such as follow-up scans, consultations, or further diagnostic procedures.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:
        
        Example Output:
        Condition Detected: Cyst

        Based on the ultrasound image, the AI model has identified the most likely condition as a Cyst, with a confidence score of 92.30%. This suggests the presence of a fluid-filled sac, which is typically benign and may not cause symptoms. Ultrasound imaging uses sound waves to create real-time visuals of internal organs and is effective in detecting such abnormalities. While most cysts are harmless, a follow-up consultation with a healthcare professional is recommended to evaluate its size, nature, and whether further investigation is needed.

        Disclaimer: This is an AI-generated summary and not a substitute for professional medical advice.

        Output the result as one clear and concise paragraph of around 100 words for easy understanding by non-medical users.
        '''
        
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
def generate_medical_report(symptoms: List[str], image_bytes: bytes, modality: str, mime_type: Optional[str] = None) -> str:
    # Prepare prompt
    template = PROMPT_TEMPLATES.get(modality.lower(), FALLBACK_TEMPLATE)
    prompt = template.format(symptoms=", ".join(symptoms))
    # prompt = (
    #     f"Based on the provided image and the following symptoms: {', '.join(symptoms)}, "
    #     "generate a clear, concise, and professional medical report. "
    #     "Include possible diagnoses, recommended next steps, and any relevant notes."
    # )
    # Wrap image bytes in Part for multimodal input
    

    contents = []
    # Only wrap image if we actually have bytes and a valid mime_type
    if image_bytes is not None and mime_type and mime_type.startswith("image/"):
        from google.genai.types import Part
        image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
        contents.append(image_part)

    # Always add the prompt
    contents.append(prompt)

    # Generate content with image part and prompt
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=contents
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
        # Extract the disease from the report
        match = re.search(r"Disease Expected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"
        # Store the report in a global variable
        latest_reports[modality] = {
        "disease": disease,
        "symptoms": symptoms,
        "report": report
        
        }
        return JSONResponse(content={"symptoms": symptoms, "disease": disease ,"report": report})
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


# CT 2D and 3D routes
@app.post("/predict/ct/2d/")
async def generate_report_ct2d(file: UploadFile = File(...)):
    modality = "ct"
    mode = "2d"

    # Only allow image files for 2D slices
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type for CT2D.")

    temp_path = f"temp_ct2d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # Inference
        raw_preds = process_ct(temp_path, mode=mode, device="cpu")
        symptoms = extract_top_symptoms(raw_preds)

        # Read image bytes before deleting temp
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        # Generate report using correct MIME type
        report = generate_medical_report(
            symptoms, img_bytes, modality=modality, mime_type=file.content_type
        )

        # Extract disease
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # Store
        latest_reports["ct2d"] = {
            "symptoms": symptoms,
            "disease": disease,
            "report": report
        }

        return JSONResponse({
            "symptoms": symptoms,
            "disease": disease,
            "report": report
        })

    except HTTPException:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))



## 3d route 
@app.post("/predict/ct/3d/")
async def generate_report_ct3d(file: UploadFile = File(...)):
    # 1) Save upload to disk
    temp_path = f"temp_ct3d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # 2) Run your 3D model to get symptoms label(s)
        raw_preds = process_ct(temp_path, mode="3d", device="cpu")
        label, prob = raw_preds[0] # type: ignore
        symptoms = [label]

        # 3) Load volume and pick mid-slices
        img = load(temp_path)
        vol = img.get_fdata() #type: ignore
        z, y, x = [d // 2 for d in vol.shape]
        slices = {
            "axial":   vol[z, :, :],
            "coronal": vol[:, y, :],
            "sagittal":vol[:, :, x],
        }

        # 4) Convert each slice to PNG bytes
        image_parts = []
        for name, sl in slices.items():
            # normalize slice to [0,255]
            sl_norm = ((sl - sl.min())/(sl.max()-sl.min()) * 255).astype(np.uint8)
            pil = Image.fromarray(sl_norm).convert("L").resize((224,224))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_parts.append(Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        os.remove(temp_path)

        # 5) Build prompt & send all three images + prompt
        prompt = (
        '''
        You are a medical AI assistant specialized in interpreting 3D and 2D CT scan results. 
        Given a set of AI-generated confidence scores for tumor detection, your task is to:

        1. Identify whether a tumor or no tumor is more likely based on the highest confidence score.
        2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 92.00%).
        3. Explain what this result means for the patient in clear, simple language.
        4. Describe briefly how 3D CT scans assist in detecting tumors by providing detailed cross-sectional views of the body.
        5. Recommend possible next steps such as further imaging or biopsy for confirmation.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:

        Output example 
        Condition Detected: Tumor
        The AI analysis of your 3D CT scan of the brain indicates a high probability of a tumor, with a confidence score of 92.00%. This suggests there may be an abnormal mass or growth
        present in the scanned region. 3D CT scans allow doctors to view detailed cross-sectional images of internal tissues, making it easier to identify potential issues like 
        tumors. While this result is a strong indicator, it is not a confirmed diagnosis. Further testing, such as an MRI or biopsy, may be required. 
        Disclaimer: This is an AI-generated summary. Please consult a certified doctor or radiologist for medical confirmation and advice.
        '''
        )

        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=[*image_parts, prompt]
        )
        report = response.text or "<empty>"
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # 6) Store & return
        latest_reports["ct3d"] = {
            "Symptom": label,
            "disease": disease,
            "report": report
        }
        return JSONResponse(latest_reports["ct3d"])

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/predict/ct/2d/")
async def get_latest_report_ct2d():
    if "ct2d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 2D CT report available.")
    return latest_reports["ct2d"]

@app.get("/predict/ct/3d/")
async def get_latest_report_ct3d():
    if "ct3d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 3D CT report available.")
    return latest_reports["ct3d"]

@app.post("/predict/mri/3d/")
async def generate_report_mri3d(file: UploadFile = File(...)):  
    # 1) Save upload to disk
    temp_path = f"temp_mri3d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        # 2) Run your 3D model to get symptoms label(s)
        raw_preds = process_mri(temp_path, mode='3d', device="cpu")
        label, prob = raw_preds[0]
        symptoms = [label]

        # 3) Load volume and pick mid-slices
        img = load(temp_path)
        vol = img.get_fdata() #type: ignore
        z, y, x = [d // 2 for d in vol.shape]
        slices = {
            "axial":   vol[z, :, :],
            "coronal": vol[:, y, :],
            "sagittal":vol[:, :, x],
        }

        # 4) Convert each slice to PNG bytes
        image_parts = []
        for name, sl in slices.items():
            # normalize slice to [0,255]
            sl_norm = ((sl - sl.min())/(sl.max()-sl.min()) * 255).astype(np.uint8)
            pil = Image.fromarray(sl_norm).convert("L").resize((224,224))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_parts.append(Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        os.remove(temp_path)

        # 5) Build prompt & send all three images + prompt
        prompt = (
            '''
                You are a medical specialist in interpreting brain MRI results. 
                Based on the image and the patient symptoms: {symptoms}, your task is to:

                1. Identify the condition with the highest confidence score from the list: ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"].
                2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 87.45%).
                3. Explain what this result means for the patient in clear, simple language, based on the detected condition.
                4. Describe briefly how brain MRI helps in identifying such conditions by providing high-resolution images of soft tissues.
                5. Suggest possible next steps, such as neurologist consultation, further imaging, or biopsy, depending on the condition.
                6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
                7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
                8. Report size should be always between 200 and 300 words.
                9. create a detailed MRI report including key findings, interpretation, and suggested follow‑up
                9. Use the following format for the output:

                Condition Detected: Glioma
                The AI analysis of your brain MRI scan suggests a high probability of Glioma, with a confidence score of 89.00%. Gliomas are tumors that originate in the glial cells of the brain or spinal cord. They can affect brain function depending on their location, size, and growth rate, potentially causing symptoms such as headaches, seizures, or neurological changes.

                MRI scans are highly effective for detecting such tumors, as they offer detailed images of soft brain tissues. This allows for accurate visualization of the tumor's structure and position, which is crucial for early diagnosis and treatment planning.

                Although this result indicates a strong likelihood of Glioma, it is not a confirmed medical diagnosis. You should consult a neurologist or oncologist for further evaluation. Additional tests like a contrast-enhanced MRI or biopsy may be recommended to validate the finding.

                Disclaimer: This is an AI-generated result. Please seek advice from a certified medical professional.
            '''
        ).format(symptoms=symptoms)


        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=[*image_parts, prompt]
        )
        report = response.text or "<empty>"
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # 6) Store & return
        latest_reports["mri3d"] = {
            "Symptom": label,
            "disease": disease,
            "report": report
        }
        return JSONResponse(latest_reports["mri3d"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/predict/mri/3d/")
async def get_latest_report_mri3d():
    if "mri3d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 3D MRI report available.")
    return latest_reports["mri3d"]

@app.post("/predict/ultrasound/")
async def generate_report_ultrasound(file: UploadFile = File(...)):
    modality = "ultrasound"

    # 1) Validate content type before saving
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # 2) Save upload to disk
    temp_path = f"temp_{modality}_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # 3) Run your ultrasound model to get symptom labels
        raw_preds = process_ultrasound(temp_path, device="cpu")
        symptoms = extract_top_symptoms(raw_preds)

        # 4) Read bytes for report generation
        with open(temp_path, "rb") as f:
            img_bytes = f.read()

        # remove temp file ASAP
        os.remove(temp_path)

        # 5) Generate the Gemini‐based medical report
        report = generate_medical_report(symptoms, img_bytes, modality=modality)

        def extract_condition(report: str) -> str:
            """
            Robustly pull the text immediately following 'Condition Detected:' 
            up to the first non‑empty line, ignoring case/extra whitespace.
            """
            if not report:
                return "Unknown"

            lower = report.lower()
            keyword = "condition detected"
            start = lower.find(keyword)
            if start == -1:
                return "Unknown"

            # Find the colon after the keyword
            colon = report.find(":", start + len(keyword))
            if colon == -1:
                return "Unknown"

            # Grab everything after the colon
            tail = report[colon+1:]

            # Split into lines, return the first non-blank one
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line

            return "Unknown"

        disease = extract_condition(report)
        # 7) Store in global for frontend polling if needed
        latest_reports[modality] = {
            "disease":  disease,
            "symptoms": symptoms,
            "report":   report,
        }

        # 8) Return JSON
        return JSONResponse(
            content={"symptoms": symptoms, "disease": disease, "report": report}
        )

    except HTTPException:
        # Already an HTTPException—nothing extra to clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        # Catch‐all: ensure temp file is removed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/predict/ultrasound/")
async def get_latest_report_ultrasound():   
    if "ultrasound" not in latest_reports:
        raise HTTPException(status_code=404, detail="No ultrasound report available.")
    return latest_reports["ultrasound"]

# Mock database of doctors
class Doctor(BaseModel):
    name: str
    specialty: str
    location: str
    phone: str
    lat: float
    lng: float

def build_overpass_query(lat: float, lng: float, shift: float = 0.03) -> str:
    lat_min = lat - shift
    lng_min = lng - shift
    lat_max = lat + shift
    lng_max = lng + shift
    return f"""
    [out:json][timeout:25];
    node
      [healthcare=doctor]
      ({lat_min},{lng_min},{lat_max},{lng_max});
    out;
    """

@app.get("/api/search-doctors")
async def search_doctors(location: str, specialty: str = ""):
    geolocator = Nominatim(user_agent="doctor-search")
    location_obj = geolocator.geocode(location + ", India")
    if not location_obj:
        return []

    lat, lon = location_obj.latitude, location_obj.longitude # type: ignore

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["healthcare"="doctor"](around:10000,{lat},{lon});
      node["amenity"="doctors"](around:10000,{lat},{lon});
    );
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(overpass_url, data=query) # type: ignore
            data = res.json()
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"detail": "Overpass API timeout. Please try again later."}
        )


    doctors = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed Doctor")
        specialty_tag = (
            tags.get("healthcare:speciality") or
            tags.get("healthcare:specialty") or
            tags.get("specialty") or
            "General"
        )
        if specialty and specialty.lower() not in specialty_tag.lower():
            continue

        phone = tags.get("phone", "Not available")
        addr = tags.get("addr:city") or tags.get("addr:suburb") or location

        doctors.append({
            "name": name,
            "specialty": specialty_tag,
            "location": addr,
            "phone": phone,
            "lat": el.get("lat"),
            "lng": el.get("lon")
        })

    return doctors
# @app.get("/api/get-doctor/{doctor_id}", response_model=Doctor)


#chatbot of landing page 

class ChatRequest(BaseModel):
    message: str

@app.post("/chat_with_report/")
async def chat_with_report(request: ChatRequest):
    user_message = request.message.lower()

    # Rule-based chatbot responses
    if "upload" in user_message and "image" in user_message:
        reply = (
            "To upload a medical image, go to the 'Upload' section from the navbar. "
            "There, you can choose from 5 model types: MRI, X-ray, Ultrasound, CT Scan 2D, and CT Scan 3D. "
            "After selecting the type and uploading your image, click 'Upload and Analyze' to get the result."
        )
    elif "analyze" in user_message or "report" in user_message:
        reply = (
            "Once you upload an image and select the model type, clicking 'Upload and Analyze' will route you to the result page. "
            "This page displays an AI-generated diagnostic report based on the image you provided."
        )
    elif "features" in user_message:
        reply = (
            "Our website offers features like disease prediction using 6 medical models, instant report generation, "
            "testimonials from patients, a FAQ section, and easy contact options."
        )
    elif "models" in user_message or "which scans" in user_message:
        reply = (
            "The supported models are:\n"
            "- MRI 2D\n- MRI 3D\n- X-ray\n- Ultrasound\n- CT Scan 2D\n- CT Scan 3D"
        )
    elif "contact" in user_message:
        reply = (
            "You can find the contact section by scrolling to the 'Contact' part of the homepage, or directly in the footer."
        )
    elif "testimonials" in user_message:
        reply = (
            "We showcase real testimonials from users who have benefited from our AI diagnosis platform."
        )
    elif "faq" in user_message or "questions" in user_message:
        reply = (
            "The FAQ section answers common questions related to uploading images, interpreting reports, and data privacy."
        )
    elif "hero" in user_message or "homepage" in user_message:
        reply = (
            "The hero section on our homepage highlights the goal of our platform — fast and accurate diagnosis from medical images using AI."
        )
    elif "cta" in user_message or "get started" in user_message:
        reply = (
            "The Call-To-Action (CTA) section encourages users to start using the platform by uploading an image and receiving a report."
        )
    else:
        reply = (
            "I'm here to help you with any questions about using the platform. "
            "You can ask me how to upload images, what models are supported, or what happens after analysis."
        )

    return {"response": reply}