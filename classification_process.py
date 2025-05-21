import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms as T

# Custom classifier model based on ResNet-50 architecture
class SkinLesionClassifier_Resnet50(nn.Module):
    def __init__(self, num_classes=7, freeze_base=True):
        """
        Initializes a ResNet-50 model for skin lesion classification.

        Args:
            num_classes (int): Number of output classes.
            freeze_base (bool): If True, freezes early layers of the base model for transfer learning.
        """
        super(SkinLesionClassifier_Resnet50, self).__init__()
        self.freeze_base = freeze_base

        # Load pre-trained ResNet-50 from torchvision with ImageNet weights
        self.base_model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

        # Optionally freeze all base model layers except the final block (layer4)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True

        # Replace the classification head with a custom fully connected classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

# Load the trained classifier model from a file
def load_model_classifier(file_name, device, num_classes=7):
    """
    Loads a trained classifier model from a file.

    Args:
        file_name (str): Path to the model .pth file.
        device (torch.device): Device to map the model to (CPU or CUDA).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = SkinLesionClassifier_Resnet50(num_classes=num_classes).to(device)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        print("Classification model loaded successfully.")
        model.eval()
    return model

# Perform inference on a list of mole crops using the trained classifier
def predict_moles(classifier, mole_crops, device):
    """
    Predicts class probabilities for each cropped mole image.

    Args:
        classifier (torch.nn.Module): Trained classification model.
        mole_crops (list): List of cropped mole images (as numpy arrays).
        device (torch.device): Device to run the inference on.

    Returns:
        list: A list of probability vectors for each mole crop.
    """
    classifier.eval()

    # Define preprocessing transform
    transform = T.Compose([
        T.ToTensor(),  # Converts PIL Image or numpy.ndarray to torch.FloatTensor and scales to [0.0, 1.0]
    ])

    predictions = []

    with torch.no_grad():
        for crop in mole_crops:

            # Show cropped image for debugging/visualization
            plt.figure(figsize=(4, 4))
            plt.imshow(crop)
            plt.title("Cropped Image")
            plt.axis("off")
            plt.show()

            # Ensure the crop is a PIL image
            if isinstance(crop, np.ndarray):
                crop = Image.fromarray(crop)

            # Apply transform and move to device
            input_tensor = transform(crop).unsqueeze(0).to(device)

            # Forward pass
            output = classifier(input_tensor)

            # Convert raw logits to probabilities using softmax
            prob = torch.softmax(output, dim=1).squeeze().tolist()
            predictions.append(prob)

    return predictions
