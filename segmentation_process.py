import numpy as np
import cv2
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms

def image_preparation_for_mask_rcnn(image, device):
    """
    Prepares an image for Mask R-CNN model (TorchVision).
    Accepts a PIL.Image or NumPy array.
    Converts the image to a tensor with shape [C, H, W] and values [0, 1].
    Returns a list of tensors (batched), as expected by Mask R-CNN.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert PIL image to NumPy array

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    transform = transforms.ToTensor()  # Convert image to tensor with values in [0, 1]
    image_tensor = transform(image).to(device)

    return [image_tensor]  # Wrap in list as expected by Mask R-CNN

def load_model_segmentation(file_name, device):
    """
    Loads a Mask R-CNN model with a ResNet50 backbone for segmentation tasks.
    Sets number of classes to 2 (background + mole).
    Loads weights from file if provided.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)
    num_classes = 2  # 1 class (mole) + background

    # Replace the box predictor with a new one for our number of classes
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # Load model weights if a path is provided
    if file_name:
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        print("Segmentation model loaded successfully.")
        model.eval()
    return model

def segment_single_image(model, image_tensor, threshold=0.9, show=True):
    """
    Runs the segmentation model on a single image tensor.
    Returns the original image (as PIL) and the predicted masks.
    Optionally shows the masks overlayed on the image.
    """
    model.eval()

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Handle batch input by extracting the first image
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension

    image_pil = F.to_pil_image(image_tensor.cpu())  # Convert tensor to PIL image

    # Extract binary masks using the confidence threshold
    masks = (prediction["masks"] > threshold).squeeze(1).cpu().numpy()

    # Optionally show the masks on the image
    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(image_pil)
        for mask in masks:
            plt.imshow(mask, alpha=0.4, cmap='jet')
        plt.title("Predicted Masks Overlay")
        plt.axis("off")
        plt.show()

    return image_pil, masks

def extract_moles(image_pil, masks, output_size=(224, 224)):
    """
    Extracts individual mole regions from an image using predicted masks.
    Returns cropped mole images, their bounding boxes, and mask crops.
    """
    image_np = np.array(image_pil)
    mole_crops = []
    bboxes = []
    mask_crops = []

    for mask in masks:
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 5 or h < 5:  # Ignore very small objects
                continue

            # Crop and resize the mole area from the image
            crop = image_np[y:y+h, x:x+w]
            resized_crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)

            # Crop and resize the mask
            crop_mask = mask[y:y+h, x:x+w].astype(np.uint8)
            resized_mask = cv2.resize(crop_mask, output_size, interpolation=cv2.INTER_NEAREST)

            mole_crops.append(resized_crop)
            bboxes.append((x, y, w, h))
            mask_crops.append(resized_mask)

    return mole_crops, bboxes, mask_crops

def apply_mask(image, mask):
    """
    Applies a binary mask to an image and returns the cropped masked region.
    """
    mask = (mask > 0.8).astype(np.uint8)  # Ensure binary mask

    masked_image = cv2.bitwise_and(image, image, mask=mask)  # Apply mask

    # Find bounding rectangle around the masked area
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the masked region
    cropped_image = masked_image[y:y + h, x:x + w]

    return cropped_image

def extract_moles_from_contours(image_pil, masks, output_size=(224, 224)):
    """
    Crops masked areas from multiple images using contours.
    Useful when masks and images are provided as lists.
    """
    cropped_images = []
    for i, (image, mask) in enumerate(zip(image_pil, masks)):
        print(f"Processing image {i+1}")

        # Apply mask to image and crop the region
        cropped_image = apply_mask(image, mask)

        cropped_images.append(cropped_image)

        # Optional visualization:
        # plt.figure(figsize=(20, 20))
        # plt.imshow(cropped_image)
        # plt.title("Cropped Image")
        # plt.axis("off")
        # plt.show()

    return cropped_images
