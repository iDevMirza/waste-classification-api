import torch
import timm

from app.config import (CLASSES, MODEL_PATH, MODEL_NAME)

class WasteClassifierModel:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)

        else:
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)

        self.model.eval()
        print(f"Model {MODEL_NAME} loaded successfully with {len(CLASSES)} classes.")


    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)
        predicted_index = predicted_index.item()

        confidence = confidence.item()

        predicted_class = CLASSES[predicted_index]

        all_probabilities = {}

        for index, class_name in enumerate(CLASSES):
            all_probabilities[class_name] = float(probabilities[0][index].item())

        return {
            "predicted_class": predicted_class,
            "predicted_class_index": predicted_index,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }

model_classifier = WasteClassifierModel()