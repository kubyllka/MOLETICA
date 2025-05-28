import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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