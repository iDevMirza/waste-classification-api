from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weights" / "EfficientViT.pt"
DATASET_PATH = BASE_DIR / "dataset"
METADATA_FILE = BASE_DIR / "metadata.csv"

CLASSES = [
    "cardboard",
    "glass",    
    "metal",
    "paper",
    "plastic",
    "trash", 
]

IMAGE_SIZE = 224
MODEL_NAME = "efficientvit_b0"
MODEL_VERSION = "efficientvit_b0_v1"