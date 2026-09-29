from PIL import Image
import torch
from torchvision import transforms
from app.config import Config as cfg

transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor