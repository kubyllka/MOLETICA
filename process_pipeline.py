import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
from PIL import Image
import albumentations as A
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
from torchvision import transforms as T
import matplotlib.pyplot as plt

# --- Models ---

class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*out_channels, 2*out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.flatten_size = 2 * out_channels * 150 * 113
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2*out_channels*150*113),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2*out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 2*self.out_channels, 150, 113)
        x = self.conv(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

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

# --- Preprocessing and utility transforms ---

class BorderCrop(A.ImageOnlyTransform):
    def __init__(self, percent: float = 0.1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.percent = percent

    def apply(self, image, **params):
        h, w = image.shape[:2]
        h_crop = int(h * self.percent)
        w_crop = int(w * self.percent)
        return image[h_crop:h - h_crop, w_crop:w - w_crop]

def psnr_numpy(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# --- OOP pipeline classes ---

class AutoencoderPipeline:
    def __init__(self, model: Autoencoder, device):
        self.model = model.to(device)
        self.device = device

    def image_preparation(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        transform = A.Compose([
            BorderCrop(percent=0.1),
            A.LongestMaxSize(max_size=597),
            A.PadIfNeeded(min_height=597, min_width=449, border_mode=cv2.BORDER_REPLICATE),
            A.Resize(height=597, width=449),
            A.ToFloat(max_value=255),
        ])
        transformed = transform(image=image)
        image_transformed = transformed['image']
        if isinstance(image_transformed, np.ndarray) and image_transformed.ndim == 3:
            tensor = torch.from_numpy(image_transformed).float().permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("Transformed image має бути 3-вимірним NumPy масивом (H, W, C)")
        return tensor.to(self.device)

    def validate(self, image):
        tensor = self.image_preparation(image)
        with torch.no_grad():
            output_tensor = self.model(tensor).detach().cpu()
        input_np = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
        H, W, C = input_np.shape
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
        mse_value = np.mean((input_np - output_np) ** 2)
        ssim_value = ssim(input_np, output_np, data_range=1.0, channel_axis=2, win_size=win_size)
        psnr_value = psnr_numpy(input_np, output_np)
        cosine_sim = cosine_similarity(
            tensor.flatten().cpu().unsqueeze(0),
            output_tensor.flatten().unsqueeze(0)
        ).item()
        return {
            "MSE": mse_value,
            "SSIM": ssim_value,
            "PSNR": psnr_value,
            "Cosine Similarity": cosine_sim
        }

class SegmentationPipeline:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def image_preparation(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")
        transform = T.ToTensor()
        image_tensor = transform(image).to(self.device)
        return [image_tensor]

    def segment(self, image, threshold=0.9, show=True):
        tensor = self.image_preparation(image)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(tensor)[0]
        if isinstance(tensor, list):
            image_tensor = tensor[0]
        else:
            image_tensor = tensor
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        image_pil = T.functional.to_pil_image(image_tensor.cpu())
        masks = (prediction["masks"] > threshold).squeeze(1).cpu().numpy()
        if show:
            plt.figure(figsize=(20, 20))
            plt.imshow(image_pil)
            for mask in masks:
                plt.imshow(mask, alpha=0.4, cmap='jet')
            plt.title("Predicted Masks Overlay")
            plt.axis("off")
            plt.show()
        return image_pil, masks

    @staticmethod
    def extract_moles(image_pil, masks, output_size=(224, 224)):
        image_np = np.array(image_pil)
        mole_crops, bboxes, mask_crops = [], [], []
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 5 or h < 5:
                    continue
                crop = image_np[y:y+h, x:x+w]
                resized_crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
                crop_mask = mask[y:y+h, x:x+w].astype(np.uint8)
                resized_mask = cv2.resize(crop_mask, output_size, interpolation=cv2.INTER_NEAREST)
                mole_crops.append(resized_crop)
                bboxes.append((x, y, w, h))
                mask_crops.append(resized_mask)
        return mole_crops, bboxes, mask_crops

    @staticmethod
    def apply_mask(image, mask):
        mask = (mask > 0.8).astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = masked_image[y:y + h, x:x + w]
        return cropped_image

    @staticmethod
    def extract_moles_from_contours(mole_crops, mask_crops, output_size=(224, 224)):
        cropped_images = []
        for image, mask in zip(mole_crops, mask_crops):
            cropped_image = SegmentationPipeline.apply_mask(image, mask)
            cropped_images.append(cropped_image)
        return cropped_images

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

# --- Model loading helpers ---

def load_model_autoencoder(file_name, device):
    encoder = Encoder(in_channels=3, out_channels=8, latent_dim=100)
    decoder = Decoder(in_channels=3, out_channels=8, latent_dim=100)
    autoencoder_model = Autoencoder(encoder, decoder).to(device)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        autoencoder_model.load_state_dict(state_dict)
        autoencoder_model.eval()
    return autoencoder_model

def load_model_classifier(file_name, device, num_classes=7):
    model = SkinLesionClassifierResnet50(num_classes=num_classes).to(device)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    return model

def load_model_segmentation(file_name, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)
    num_classes = 2
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    return model