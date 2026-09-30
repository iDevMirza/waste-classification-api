# ♻️ Waste Classification API

A FastAPI-based backend for **real-time waste classification and human-in-the-loop dataset collection** using a lightweight deep learning model.

The API is designed to connect with a mobile application where users capture waste images, receive an AI-generated classification, verify the prediction, and save the image into a growing human-validated dataset.

## 📌 Project Overview

This project provides the backend API for a smart waste classification and dataset collection system.

The current system supports six waste categories:

- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

The API uses a trained **EfficientViT** model for image classification and provides endpoints for prediction, human validation, dataset collection, and dataset statistics.

The long-term objective is to create a continuously growing real-world waste dataset that can be used to analyse model errors and retrain improved waste classification models.

---

## ✨ Features

- Real-time waste image classification
- EfficientViT-based lightweight deep learning model
- Six-class waste classification
- Prediction confidence scores
- Individual class probabilities
- Human-in-the-loop prediction validation
- Automatic image organisation by confirmed class
- Dataset metadata generation
- Model error tracking
- Dataset class statistics
- Interactive Swagger API documentation
- Designed for integration with mobile applications

---

## 🧠 Supported Classes

The model currently classifies images into:

```text
cardboard
glass
metal
paper
plastic
trash
```

---

## 🔄 System Workflow

```text
Mobile Application
        │
        │ Capture Waste Image
        ▼
     FastAPI
        │
        │ POST /predict
        ▼
 EfficientViT Model
        │
        ▼
Prediction + Confidence
        │
        ▼
   Mobile Application
        │
        │ User reviews prediction
        ▼
Is the prediction correct?
        │
     ┌──┴──┐
    YES    NO
     │      │
     │      └── Select correct class
     │
     ▼
Human-Confirmed Label
        │
        │ POST /dataset/save
        ▼
Human-Validated Dataset
        │
        ├── Images
        └── Metadata
```

---

## 👤 Human-in-the-Loop Dataset Collection

A key feature of this project is that model predictions are not automatically treated as ground truth.

After receiving a prediction, the user verifies whether the classification is correct.

### Correct Prediction

For example:

```text
Model Prediction: plastic
Confidence: 95.32%

Human Confirmation: YES
```

The image is saved to:

```text
dataset/plastic/
```

### Incorrect Prediction

For example:

```text
Model Prediction: plastic
Confidence: 81.20%

Human Confirmation: NO

Correct Label: glass
```

The image is saved to:

```text
dataset/glass/
```

The original model prediction is still recorded in the metadata, allowing model mistakes to be analysed later.

---

## 📁 Project Structure

```text
waste_classification_api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── dataset_manager.py
│   └── config.py
│
├── weights/
│   └── EfficientViT.pt
│
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
|
├── metadata.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies

The project is built using:

- **Python**
- **FastAPI**
- **PyTorch**
- **timm**
- **EfficientViT**
- **Pillow**
- **NumPy**
- **Uvicorn**

The API is designed to be integrated with a mobile frontend such as **Flutter**.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project:

```bash
cd waste_classification_api
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Model Setup

Place the trained EfficientViT model weights inside:

```text
weights/
```

For example:

```text
weights/EfficientViT.pt
```

The model architecture, number of classes, class order, preprocessing pipeline, and checkpoint format must match the configuration used during model training.

> **Note:** Large trained model files may be excluded from the repository using `.gitignore`. In that case, the model weights must be downloaded or added separately.

---

## ▶️ Running the API

Start the development server with:

```bash
uvicorn app.main:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

---

## 📖 Swagger Documentation

FastAPI automatically generates interactive Swagger documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

Swagger can be used to test all API endpoints before integrating the backend with the mobile application.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Check API and model status |
| `POST` | `/predict` | Classify an uploaded waste image |
| `POST` | `/dataset/save` | Save a human-validated image |
| `GET` | `/dataset/stats` | Retrieve dataset statistics |

---

## 🔍 Prediction API

### Endpoint

```http
POST /predict
```

### Input

The endpoint accepts an image file using `multipart/form-data`.

### Example Response

```json
{
  "success": true,
  "model": "efficientvit_b0_v1",
  "prediction": "plastic",
  "class_index": 4,
  "confidence": 0.9532,
  "probabilities": {
    "cardboard": 0.0021,
    "glass": 0.0102,
    "metal": 0.0045,
    "paper": 0.0032,
    "plastic": 0.9532,
    "trash": 0.0268
  }
}
```

---

## 💾 Dataset Save API

### Endpoint

```http
POST /dataset/save
```

The endpoint accepts:

```text
file
model_prediction
confidence
human_label
```

If the model prediction is correct:

```text
model_prediction = plastic
human_label = plastic
```

If the model prediction is incorrect:

```text
model_prediction = plastic
human_label = glass
```

The image is always stored according to the **human-confirmed label**.

### Example Response

```json
{
  "success": true,
  "message": "Image saved to dataset.",
  "filename": "glass_000001.jpg",
  "label": "glass",
  "model_prediction": "plastic",
  "confidence": 0.81,
  "prediction_correct": false,
  "path": "glass/glass_000001.jpg"
}
```

---

## 📊 Dataset Statistics

### Endpoint

```http
GET /dataset/stats
```

### Example Response

```json
{
  "total_images": 150,
  "classes": {
    "cardboard": 24,
    "glass": 27,
    "metal": 21,
    "paper": 25,
    "plastic": 31,
    "trash": 22
  }
}
```

This endpoint can be used by the mobile application or a future dashboard to monitor dataset growth and class distribution.

---

## 🗂️ Dataset Organisation

Human-validated images are automatically organised into class folders:

```text
dataset/
│
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

For example:

```text
dataset/plastic/
├── plastic_000001.jpg
├── plastic_000002.jpg
└── plastic_000003.jpg
```

---

## 📑 Metadata

Alongside the images, the API maintains:

```text
metadata.csv
```

Example:

```csv
filename,model_prediction,confidence,human_label,timestamp,model_version
plastic_000001.jpg,plastic,0.9532,plastic,2026-09-30T10:20:30+00:00,efficientvit_b0_v1
glass_000001.jpg,plastic,0.8120,glass,2026-09-30T10:22:15+00:00,efficientvit_b0_v1
```

This allows researchers to retain both the original model prediction and the final human-confirmed label.

The metadata can later be used for:

- Error analysis
- Confusion analysis
- Confidence analysis
- Dataset quality assessment
- Model evaluation
- Model retraining

---

## 🔁 Model Improvement Pipeline

The project is designed to support an iterative machine learning workflow:

```text
Initial Model
     ↓
Mobile Deployment
     ↓
Real-World Image Collection
     ↓
AI Classification
     ↓
Human Validation
     ↓
New Labelled Dataset
     ↓
Error Analysis
     ↓
Model Retraining
     ↓
Improved Model
     ↓
Redeployment
```

This creates a practical feedback loop between **model deployment, human validation, dataset creation, and model improvement**.

---

## 📱 Mobile Application Integration

The intended mobile workflow is:

```text
Open App
   ↓
Capture Waste Image
   ↓
POST /predict
   ↓
Display Prediction
   ↓
"Save it in dataset"
   ↓
"Is the prediction correct?"
   ↓
YES ──────────────┐
                  │
NO → Select Class │
                  ↓
         POST /dataset/save
                  ↓
          Dataset Updated
```

The mobile frontend can be developed using Flutter and communicate with this API through HTTP requests.

---

## 🔮 Future Development

Planned extensions include:

- Flutter mobile application integration
- Camera-based real-time classification
- Improved dataset management
- Dataset balancing analysis
- Model performance monitoring
- Model error visualisation
- Automatic confusion matrix generation
- Research dashboard
- Authentication for dataset contributors
- Cloud deployment
- Database integration
- Model version management
- Retraining with newly collected data
- On-device inference experiments

---

## 🎯 Research Motivation

Waste classification models are often trained using controlled datasets that may not fully represent real-world environmental conditions.

This project explores a practical approach where a lightweight classification model is deployed as part of a mobile data collection system. Users can validate predictions while collecting new waste images, producing a growing human-confirmed dataset.

The collected data can subsequently support model evaluation, error analysis, retraining, and further research into lightweight AI for smart recycling systems.

---

## ⚠️ Project Status

This project is currently under active research and development.

The API, model configuration, dataset structure, and mobile integration may change as the research progresses.

---

## 🤝 Contributions

Contributions, suggestions, and research collaborations are welcome.

If you identify a bug or have an idea for improving the system, please open an issue or submit a pull request.

---

## 📄 License

A license has not yet been specified for this research project.

Before publicly distributing the dataset, model weights, or software for reuse, appropriate licensing and dataset usage conditions should be defined.

---

## 👨‍💻 Author

**Mirza Mahmud Hossan**

MSc Advanced Computer Science  
Cardiff Metropolitan University

Research interests include:

**Machine Learning • Deep Learning • Computer Vision • Medical Imaging • Applied Artificial Intelligence**