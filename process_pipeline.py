import pandas as pd
import numpy as np
import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import wandb
# from torchsummary import summary
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
from PIL import Image
import albumentations as A
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
import math
import torchvision.models as models
from torchvision import transforms as T
from torchvision import transforms


class Encoder( nn.Module ):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):  # Зменшено кількість каналів і латентний розмір
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size=3, padding=1 ),  # (597, 449)
            nn.ReLU(),
            nn.Conv2d( out_channels, out_channels, kernel_size=3, padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=3, stride=2, padding=1 ),  # (298, 225)
            nn.Conv2d( out_channels, 2 * out_channels, kernel_size=3, padding=1 ),
            nn.ReLU(),
            nn.Conv2d( 2 * out_channels, 2 * out_channels, kernel_size=3, padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=3, stride=2, padding=1 ),  # (150, 113)
        )

        self.flatten_size = 2 * out_channels * 150 * 113  # Оновлений розмір після MaxPool2d

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear( self.flatten_size, latent_dim ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net( x )
        x = self.fc( x )
        return x


class Decoder( nn.Module ):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):  # Зменшено кількість каналів і латентний розмір
        super().__init__()
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear( latent_dim, 2 * out_channels * 150 * 113 ),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d( 2 * out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0 ),
            # (300, 225)
            nn.ReLU(),
            nn.ConvTranspose2d( out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0 ),
            # (600, 450)
            nn.ReLU(),
            nn.ConvTranspose2d( out_channels, in_channels, kernel_size=3, padding=1 ),
            nn.Sigmoid(),  # Normalization output [0, 1]
        )

    def forward(self, x):
        x = self.fc( x )
        x = x.view( -1, 2 * self.out_channels, 150, 113 )
        x = self.conv( x )
        return x


class Autoencoder( nn.Module ):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.encoder = encoder
        self.encoder.to( device )
        self.decoder = decoder
        self.decoder.to( device )

    def forward(self, x):
        encoded = self.encoder( x )
        decoded = self.decoder( encoded )
        return decoded

# Custom transformation for cropping borders
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
    PIXEL_MAX = 1.0  # Assuming images are normalized to [0,1]
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def image_preparation(image, device):
    # image_bgr = cv2.imread(image_name)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # img_rotated = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    # show_image(img_rotated)

    if isinstance( image, Image.Image ):
        image = np.array( image )

    transform_validation = A.Compose([
        BorderCrop(percent=0.1),  # підрізаємо краї на 10%
        A.LongestMaxSize(max_size=597),  # зберігаємо пропорції
        A.PadIfNeeded(
            min_height=597, min_width=449,
            border_mode=cv2.BORDER_REPLICATE  # дублюємо краї, а не додаємо чорний
        ),
        A.Resize(height=597, width=449),
        A.ToFloat(max_value=255),
    ])

    transformed = transform_validation(image=image)
    image_transformed = transformed['image']

    if isinstance(image_transformed, np.ndarray) and image_transformed.ndim == 3:
        test_image_tensor = torch.from_numpy(image_transformed).float().permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Transformed image має бути 3-вимірним NumPy масивом (H, W, C)")

    return test_image_tensor.to(device)

from PIL import Image
import torch
import numpy as np
from torchvision import transforms

def image_preparation_for_mask_rcnn(image, device):
    """
    Prepares an image for Mask R-CNN model (TorchVision).
    Accepts a PIL.Image or NumPy array.
    Converts the image to a tensor with shape [C, H, W] and values [0, 1].
    Returns a list of tensors (batched), as expected by Mask R-CNN.
    """
    # If the image is in PIL format, convert it to NumPy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure image is RGB (3 channels)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Convert the image to a float tensor and scale values to [0, 1]
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)

    # Return the image as a list of tensors (single image) for Mask R-CNN
    return [image_tensor]  # Returning a list with one 3D tensor


class SkinLesionClassifier_Resnet50( nn.Module ):
    def __init__(self, num_classes=7, freeze_base=True):
        super( SkinLesionClassifier_Resnet50, self ).__init__()
        self.freeze_base = freeze_base

        self.base_model = models.resnet50( weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 )

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear( num_features, 512 ),
            nn.ReLU(),
            nn.Dropout( 0.2 ),
            nn.Linear( 512, num_classes ),
        )

    def forward(self, x):
        return self.base_model( x )

def load_model_autoencoder(file_name, device):
    encoder = Encoder(in_channels=3, out_channels=8, latent_dim=100)
    decoder = Decoder(in_channels=3, out_channels=8, latent_dim=100)
    autoencoder_model = Autoencoder(encoder, decoder, device)
    autoencoder_model.to(device)

    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        autoencoder_model.load_state_dict(state_dict)
        print("Autoencoder model state loaded successfully.")
        autoencoder_model.eval()

    return autoencoder_model

def load_model_classifier(file_name, device, num_classes=7):
    model = SkinLesionClassifier_Resnet50(num_classes=num_classes).to(device)
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        print("Classification model loaded successfully.")
        model.eval()
    return model

def load_model_segmentation(file_name, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)
    num_classes = 2  # 1 клас (родимка) + фон

    # Оновлюємо box_predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Оновлюємо mask_predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        print("Segmentation model loaded successfully.")
        model.eval()
    return model
def validate_image(autoencoder_model, test_image_tensor, device):
    if isinstance(test_image_tensor, np.ndarray):
        test_image_tensor = torch.from_numpy(test_image_tensor).float()

    test_image_tensor = test_image_tensor.to(device)

    with torch.no_grad():
        output_tensor = autoencoder_model(test_image_tensor).detach().cpu()  # (1, C, H, W)

    # Підготовка до метрик — перетворимо на NumPy
    input_np = test_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()           # (H, W, C)

    H, W, C = input_np.shape
    win_size = min(H, W)
    if win_size % 2 == 0:
        win_size -= 1  # ssim win_size має бути непарним

    mse_value = np.mean((input_np - output_np) ** 2)
    ssim_value = ssim(input_np, output_np, data_range=1.0, channel_axis=2, win_size=win_size)
    psnr_value = psnr_numpy( input_np, output_np )

    cosine_sim = cosine_similarity(
        test_image_tensor.flatten().cpu().unsqueeze(0),
        output_tensor.flatten().unsqueeze(0)
    ).item()

    return {
        "MSE": mse_value,
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "Cosine Similarity": cosine_sim
    }

def segment_single_image(model, image_tensor, threshold=0.9, show=True):
    model.eval()

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Якщо image_tensor є списком, то це може бути результатом пакетної обробки
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]  # Беремо перший елемент списку

    # Перевірка форми тензора
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Якщо це батч, видаляємо перший вимір

    # Перетворюємо зображення в PIL
    image_pil = F.to_pil_image(image_tensor.cpu())

    # Отримуємо маски
    masks = (prediction["masks"] > threshold).squeeze(1).cpu().numpy()

    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(image_pil)
        for mask in masks:
            plt.imshow(mask, alpha=0.4, cmap='jet')  # Накладаємо кожну маску
        plt.title("Predicted Masks Overlay")
        plt.axis("off")
        plt.show()

    return image_pil, masks


def extract_moles(image_pil, masks, output_size=(224, 224)):
    image_np = np.array(image_pil)
    mole_crops = []
    bboxes = []
    mask_crops = []

    for mask in masks:
        mask_uint8 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 5 or h < 5:
                continue

            # Crop the mole from the image
            crop = image_np[y:y+h, x:x+w]
            resized_crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)

            # Crop the corresponding mask and convert it to uint8
            crop_mask = mask[y:y+h, x:x+w].astype(np.uint8)
            resized_mask = cv2.resize(crop_mask, output_size, interpolation=cv2.INTER_NEAREST)

            mole_crops.append(resized_crop)
            bboxes.append((x, y, w, h))
            mask_crops.append(resized_mask)

    return mole_crops, bboxes, mask_crops

def apply_mask(image, mask):
    """
    Crops the image area corresponding to the mask.

    image: the skin lesion image
    mask: the segmentation mask

    Returns the cropped area.
    """
    # Mask should be binary (black and white)
    mask = (mask > 0.8).astype( np.uint8 )

    # Use the mask to crop only the relevant part of the image
    masked_image = cv2.bitwise_and( image, image, mask=mask )

    # Define the area where the mask is not empty
    coords = cv2.findNonZero( mask )
    x, y, w, h = cv2.boundingRect( coords )  # Find the bounding rectangle

    # Crop the image based on the bounding box
    cropped_image = masked_image[y:y + h, x:x + w]

    return cropped_image

def extract_moles_from_contours(image_pil, masks, output_size=(224, 224)):
    cropped_images = []
    for i, (image, mask) in enumerate(zip(image_pil, masks)):
        print(f"Обробляємо зображення {i+1}")

        # Накладаємо маску на зображення (можна використовувати cv2)
        cropped_image = apply_mask(image, mask)

        # Додаємо вирізане зображення до списку
        cropped_images.append(cropped_image)

        # plt.figure( figsize=(20, 20) )
        # plt.imshow(cropped_image)
        # plt.title( "cropped_image" )
        # plt.axis( "off" )
        # plt.show()

    return cropped_images

def predict_moles(classifier, mole_crops, device):
    # Переведемо модель в режим оцінки
    classifier.eval()

    # Трансформація для підготовки зображення
    transform = T.Compose([
        T.ToTensor(),
    ])

    predictions = []
    with torch.no_grad():
        for crop in mole_crops:

            plt.figure( figsize=(20, 20) )
            plt.imshow(crop)
            plt.title( "cropped_image" )
            plt.axis( "off" )
            plt.show()

            # Конвертуємо зображення в PIL, якщо воно в NumPy масиві
            if isinstance(crop, np.ndarray):
                crop = Image.fromarray(crop)

            # Застосовуємо трансформації до кожної родимки
            input_tensor = transform(crop).unsqueeze(0).to(device)  # Додаємо batch size

            # Передбачення для родимки
            output = classifier(input_tensor)

            # Витягуємо передбачення (клас з найвищою ймовірністю)
            _, predicted_class = torch.max(output, 1)
            predictions.append(predicted_class.item())

    return predictions


