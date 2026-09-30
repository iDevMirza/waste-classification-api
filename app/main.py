from io import BytesIO
from fastapi import (FastAPI, File, UploadFile, HTTPException, Form,)
from PIL import Image, UnidentifiedImageError
from app.config import (CLASSES, MODEL_VERSION,)
from app.preprocessing import preprocess_image
from app.model import model_classifier
from app.dataset_manager import (save_image, initialize_dataset, initialize_metadata, get_dataset_statistics)

app = FastAPI(
    title="Waste Classification API",
    description="AI-powered waste classification API that predicts the type of waste in an image and allows users to save images with human labels for dataset expansion.",
    version="1.0.0",
)

initialize_dataset()
initialize_metadata()

@app.get("/")
def root():
    return {
        "message": "Waste Classification API is running",
        "model": MODEL_VERSION,
        "classes": CLASSES,
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_VERSION,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image.load()  # Ensure the image is fully loaded
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

    image_tensor = preprocess_image(image)
    prediction_result = model_classifier.predict(image_tensor)

    return {
        "success": True,
        "model": MODEL_VERSION,
        "prediction": prediction_result["predicted_class"],
        "class_index": prediction_result["predicted_class_index"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["all_probabilities"],
    }

@app.post("/dataset/save")
async def save_to_dataset(
    file: UploadFile = File(...),
    human_label: str = Form(...),
    model_prediction: str = Form(...),
    confidence: float = Form(...),
):
    if model_prediction not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid model prediction: {model_prediction}. Must be one of {CLASSES}")

    if human_label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid human label: {human_label}. Must be one of {CLASSES}")

    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")

    if not file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image.load()

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

    prediction_correct = (model_prediction == human_label)

    try:
        result = save_image(image=image, human_label=human_label, model_prediction=model_prediction, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image to dataset: {str(e)}")

    return {
        "success": True,
        "message": "Image saved to dataset successfully",
        "file_name": result["file_name"],
        "label": result["label"],
        "model_prediction": (result["model_prediction"]),
        "confidence": (result["confidence"]),
        "prediction_correct": (prediction_correct),
        "timestamp": result["timestamp"],
        "path": result["path"],
    }

@app.get("/dataset/stats")
def dataset_statistics():

    return get_dataset_statistics()