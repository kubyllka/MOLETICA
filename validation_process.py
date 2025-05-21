import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
import albumentations as A
from PIL import Image

# Encoder module for the autoencoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):  # Reduced number of channels and latent size
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # (597, 449)
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (298, 225)
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (150, 113)
        )

        self.flatten_size = 2 * out_channels * 150 * 113  # Updated size after MaxPool2d

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


# Decoder module for the autoencoder
class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, latent_dim=64):  # Reduced number of channels and latent size
        super().__init__()
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2 * out_channels * 150 * 113),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2 * out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0),  # (300, 225)
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0),  # (600, 450)
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalization output [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 2 * self.out_channels, 150, 113)
        x = self.conv(x)
        return x


# Wrapper for encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)
        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Custom transformation for cropping image borders
class BorderCrop(A.ImageOnlyTransform):
    def __init__(self, percent: float = 0.1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.percent = percent

    def apply(self, image, **params):
        h, w = image.shape[:2]
        h_crop = int(h * self.percent)
        w_crop = int(w * self.percent)
        return image[h_crop:h - h_crop, w_crop:w - w_crop]


# PSNR metric calculation
def psnr_numpy(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Assuming images are normalized to [0,1]
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# Prepare input image for autoencoder
def image_preparation(image, device):
    if isinstance(image, Image.Image):
        image = np.array(image)

    transform_validation = A.Compose([
        BorderCrop(percent=0.1),  # crop 10% from borders
        A.LongestMaxSize(max_size=597),  # preserve aspect ratio
        A.PadIfNeeded(
            min_height=597, min_width=449,
            border_mode=cv2.BORDER_REPLICATE  # replicate border pixels, don't add black
        ),
        A.Resize(height=597, width=449),
        A.ToFloat(max_value=255),
    ])

    transformed = transform_validation(image=image)
    image_transformed = transformed['image']

    if isinstance(image_transformed, np.ndarray) and image_transformed.ndim == 3:
        test_image_tensor = torch.from_numpy(image_transformed).float().permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Transformed image must be a 3-dimensional NumPy array (H, W, C)")

    return test_image_tensor.to(device)


# Load autoencoder model from file
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


# Validate image and compute reconstruction metrics
def validate_image(autoencoder_model, test_image_tensor, device):
    if isinstance(test_image_tensor, np.ndarray):
        test_image_tensor = torch.from_numpy(test_image_tensor).float()

    test_image_tensor = test_image_tensor.to(device)

    with torch.no_grad():
        output_tensor = autoencoder_model(test_image_tensor).detach().cpu()  # (1, C, H, W)

    # Convert tensors to NumPy arrays for metric calculations
    input_np = test_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()           # (H, W, C)

    H, W, C = input_np.shape
    win_size = min(H, W)
    if win_size % 2 == 0:
        win_size -= 1  # ssim win_size must be odd

    mse_value = np.mean((input_np - output_np) ** 2)
    ssim_value = ssim(input_np, output_np, data_range=1.0, channel_axis=2, win_size=win_size)
    psnr_value = psnr_numpy(input_np, output_np)

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
