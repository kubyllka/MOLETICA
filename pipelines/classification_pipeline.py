import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
import torchvision

def load_model_classifier(file_name, device, num_classes=7):
    model = SkinLesionClassifierResnet50(num_classes=num_classes).to(device)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    return model
class ClassificationPipeline:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.transform = T.Compose([T.ToTensor()])

    def predict(self, mole_crops):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for crop in mole_crops:
                if isinstance(crop, np.ndarray):
                    crop = Image.fromarray(crop)
                input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                prob = torch.softmax(output, dim=1).squeeze().tolist()
                predictions.append(prob)
        return predictions

class SkinLesionClassifierResnet50(nn.Module):
    def __init__(self, num_classes=7, freeze_base=True):
        super().__init__()
        self.base_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)
