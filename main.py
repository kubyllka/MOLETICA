import json
from urllib.parse import urlparse

from process_pipeline import *
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from PIL import Image
import requests
from io import BytesIO

from config import CLASS_LABELS, DETECTION_TYPE, RESULTS_FOLDER

# To run this app:
# fastapi dev main.py

random_state = 4747
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

validation_model = load_model_autoencoder("models/autoencoder.pth", device)

classification_model = load_model_classifier("models/classifier.pth", device)

segmentation_model = load_model_segmentation("models/segmentation.pth", device)

class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")

@app.post("/validate-skin")
async def validate_skin(request: ValidationRequestModel):
    url = request.url
    print(url)

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open( BytesIO( file_content ) ).convert( "RGB" )
        img_preprocessed = image_preparation( img, device )
        results = validate_image( validation_model, img_preprocessed, device )

        is_mole = bool( results['PSNR'] >= 24 )

        return {"value": is_mole}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ClassificationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(
        ..., description="Parameter to provide a timestamp of request."
    )


@app.post("/classify-skin")
async def classify_skin(request: ClassificationRequestModel):
    url = request.url
    user_id = request.user_id
    timestamp = request.timestamp

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content)).convert("RGB")
        img_tensor = image_preparation_for_mask_rcnn( img, device )

        image_pil, masks = segment_single_image(segmentation_model, img_tensor)
        mole_crops, bboxes, mask_crops = extract_moles( image_pil, masks )
        mole_crops_from_contours = extract_moles_from_contours( mole_crops, mask_crops )

        predictions = predict_moles(classification_model, mole_crops_from_contours, device )
        return {"predictions": predictions}

        #
        # with torch.no_grad():
        #     output = classification_model(img_preprocessed)
        #     probabilities = output[0].cpu().numpy()
        #
        # result = {CLASS_LABELS[i]: float( prob ) for i, prob in enumerate( probabilities )}
        # base_path, old_folder, file_name = url.rsplit("/", 2)
        # new_file_name = f"{user_id}_{DETECTION_TYPE}_{timestamp}.txt"
        # new_s3_path = f"{base_path}/{RESULTS_FOLDER}/{new_file_name}"
        # s3_key = "/".join(new_s3_path.split("/")[3:])
        #
        # s3.put_object(
        #     Bucket=bucket_name,
        #     Key=s3_key,
        #     Body=json.dumps({**result, "image_url": url}),
        #     ContentType="application/json",
        # )
        # return {**result, "path": new_s3_path}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
