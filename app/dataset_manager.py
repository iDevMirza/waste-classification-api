from pathlib import Path
from datetime import datetime, timezone
import csv

from app.config import (
    CLASSES,
    DATASET_PATH,
    METADATA_FILE,
    MODEL_VERSION,
)

def initialize_dataset():
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        class_directory = (DATASET_PATH / class_name)
        class_directory.mkdir(parents=True, exist_ok=True)

def initialize_metadata():
    if METADATA_FILE.exists():
        return

    with open(METADATA_FILE, 'w', newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "model_prediction", "confidence", "human_label", "timestamp", "model_version"])

def generate_filename(label: str):
    class_directory = (DATASET_PATH / label)

    existing_files = list(class_directory.glob("*.jpg"))
    next_number = len(existing_files) + 1
    filename = (f"{label}_{next_number:06d}.jpg")
    return filename

def save_metadata(filename, model_prediction, confidence, human_label):
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(METADATA_FILE, 'a', newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([filename, model_prediction, confidence, human_label, timestamp, MODEL_VERSION])

def save_image(image, human_label, model_prediction, confidence):
    if human_label not in CLASSES:
        raise ValueError(f"Invalid human label: {human_label}. Must be one of {CLASSES}")

    destination_directory = (DATASET_PATH / human_label)
    destination_directory.mkdir(parents=True, exist_ok=True)

    filename = generate_filename(human_label)
    destination_path = (destination_directory / filename)

    image.save(destination_path, format="JPEG", quality=95)

    save_metadata(filename = filename, model_prediction = model_prediction, confidence = confidence, human_label = human_label)

    return {
        "file_name": filename,
        "label": human_label,
        "model_prediction": model_prediction,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": (f"{destination_path}"),
    }

def get_dataset_statistics():
    statistics = {}
    total = 0

    for class_name in CLASSES:
        class_directory = (DATASET_PATH / class_name)
        count = len(list(class_directory.glob("*.jpg")))
        statistics[class_name] = count
        total += count

    return {
        "total_images": total,
        "class_distribution": statistics
    }