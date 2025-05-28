import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import albumentations as A
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity

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

def psnr_numpy(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

class BorderCrop(A.ImageOnlyTransform):
    def __init__(self, percent: float = 0.1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.percent = percent

    def apply(self, image, **params):
        h, w = image.shape[:2]
        h_crop = int(h * self.percent)
        w_crop = int(w * self.percent)
        return image[h_crop:h - h_crop, w_crop:w - w_crop]

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

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



def load_autoencoder(path: str, device):
    encoder = Encoder(in_channels=3, out_channels=8, latent_dim=100)
    decoder = Decoder(in_channels=3, out_channels=8, latent_dim=100)
    model = Autoencoder(encoder, decoder).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
