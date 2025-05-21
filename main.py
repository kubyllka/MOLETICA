import json
import logging
from datetime import datetime
from urllib.parse import urlparse
from io import BytesIO

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from validation_process import *
from segmentation_process import *
from classification_process import *
from config import CLASS_LABELS, DETECTION_TYPE, RESULTS_FOLDER, PHOTOS_FOLDER

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_state = 4747

# Load models
validation_model = load_model_autoencoder("models/autoencoder.pth", device)
classification_model = load_model_classifier("models/classifier.pth", device)
segmentation_model = load_model_segmentation("models/segmentation.pth", device)

app = FastAPI()

# Singleton pattern for S3 client to reuse the same instance
class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance

# Pydantic models for API request validation
class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="S3 URL of the image to validate.")

class ClassificationRequestModel(BaseModel):
    url: str = Field(..., description="S3 URL of the image to classify.")
    user_id: str = Field(..., description="User identifier.")
    timestamp: str = Field(..., description="Timestamp of the request (ISO format).")

# Helper function to download image from S3
def fetch_image_from_s3(s3_url: str):
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError("URL must start with s3://")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = S3ClientSingleton()

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        file_content = response["Body"].read()
        logger.info(f"Successfully fetched image from S3: {bucket}/{key}")
        return Image.open(BytesIO(file_content)).convert("RGB"), bucket
    except Exception as e:
        logger.error(f"Failed to fetch image from S3: {e}")
        raise

@app.post("/validate-skin")
async def validate_skin(request: ValidationRequestModel):
    """
    Validate if the image likely contains a mole using an autoencoder model.
    """
    try:
        image, _ = fetch_image_from_s3(request.url)
        preprocessed = image_preparation(image, device)
        result = validate_image(validation_model, preprocessed, device)

        is_mole = result['PSNR'] >= 24
        logger.info(f"Validation PSNR: {result['PSNR']} - Is mole: {is_mole}")
        return {"value": bool(is_mole)}
    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-skin")
async def classify_skin(request: ClassificationRequestModel):
    """
    Detect, crop, and classify moles in an image using segmentation and classification models.
    Save cropped images and results back to S3.
    """
    try:
        # Parse input
        timestamp = datetime.fromisoformat(request.timestamp)
        date_str = timestamp.strftime("%Y-%m-%d")

        image, bucket = fetch_image_from_s3(request.url)
        image_tensor = image_preparation_for_mask_rcnn(image, device)

        # Segmentation stage
        segmented_image, masks = segment_single_image(segmentation_model, image_tensor)
        mole_crops, bboxes, mask_crops = extract_moles(segmented_image, masks)
        mole_crops_processed = extract_moles_from_contours(mole_crops, mask_crops)

        # Classification stage
        predictions = predict_moles(classification_model, mole_crops_processed, device)

        s3 = S3ClientSingleton()
        all_results = []

        for i, (crop, bbox, prediction) in enumerate(zip(mole_crops_processed, bboxes, predictions)):
            image_filename = f"{request.user_id}_{DETECTION_TYPE}_{request.timestamp}_{i}.jpg"
            crop_path = f"{request.user_id}/{date_str}/{PHOTOS_FOLDER}/{image_filename}"

            # Save image to S3
            buffer = BytesIO()
            Image.fromarray(crop).save(buffer, format="JPEG")
            buffer.seek(0)
            s3.put_object(Bucket=bucket, Key=crop_path, Body=buffer, ContentType="image/jpeg")
            logger.info(f"Saved crop image to S3: {crop_path}")

            # Format result entry
            x1, y1, x2, y2 = bbox
            x, y = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)

            label_probs = {CLASS_LABELS[j]: float(prob) for j, prob in enumerate(prediction)}
            result_entry = {
                "bbox": {"x": x, "y": y, "width": width, "height": height},
                "label_probabilities": label_probs,
                "image_url": f"s3://{bucket}/{crop_path}"
            }
            all_results.append(result_entry)

        # Save all results as JSON
        results_key = f"{request.user_id}/{date_str}/{RESULTS_FOLDER}/{request.user_id}_{DETECTION_TYPE}_{request.timestamp}.json"
        s3.put_object(
            Bucket=bucket,
            Key=results_key,
            Body=json.dumps({"results": all_results}, ensure_ascii=False),
            ContentType="application/json"
        )
        logger.info(f"Saved results JSON to S3: {results_key}")

        return {
            "path": f"s3://{bucket}/{results_key}",
            "results": all_results,
        }

    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(e))
