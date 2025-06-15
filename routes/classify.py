# routes/classify.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from utils.logger import logger
from utils.s3 import S3Client
from pipelines.segmentation_pipeline import SegmentationPipeline, load_model_segmentation
from pipelines.classification_pipeline import ClassificationPipeline, load_model_classifier
from config import CLASS_LABELS, PHOTOS_FOLDER, RESULTS_FOLDER, DETECTION_TYPE
import json
from datetime import datetime
from io import BytesIO
import torch

from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import HTTPException
from pydantic import BaseModel
from PIL import Image

router = APIRouter()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_state = 4747

classification_model = load_model_classifier("models/classifier.pth", device)
segmentation_model = load_model_segmentation("models/segmentation.pth", device)

segmentation_pipeline = SegmentationPipeline(segmentation_model, device)
classification_pipeline = ClassificationPipeline(classification_model, device)

class ClassifyRequest(BaseModel):
    url: str = Field( ..., description="S3 URL of the image to classify." )
    user_id: str = Field( ..., description="User identifier." )
    timestamp: str = Field( ..., description="Timestamp of the request (ISO format)." )

@router.post("/classify-moles")
async def classify(request: ClassifyRequest):
    """
        Detect, crop, and classify moles in an image using segmentation and classification models.
        Save cropped images and results back to S3.
        """
    s3 = S3Client()
    try:
        # Parse and validate timestamp
        try:
            timestamp = datetime.fromisoformat(request.timestamp )
        except ValueError:
            raise HTTPException( status_code=400, detail="Invalid timestamp format, expected ISO 8601" )

        date_str = timestamp.strftime( "%Y-%m-%d" )

        # Fetch image from S3
        try:
            image, bucket = s3.fetch_image( request.url )
        except HTTPException:
            # propagate HTTPExceptions from fetch (400, 401, 404)
            raise
        except Exception as e:
            logger.error( f"Failed to fetch image for classification: {e}" )
            raise HTTPException( status_code=500, detail="Failed to download image" )

        # Segmentation stage
        segmented_image, masks = segmentation_pipeline.segment( image )
        mole_crops, bboxes, mask_crops = segmentation_pipeline.extract_moles( segmented_image, masks )
        mole_crops_processed = segmentation_pipeline.extract_moles_from_contours( mole_crops, mask_crops )

        # Classification stage
        predictions = classification_pipeline.predict( mole_crops_processed )

        all_results = []

        for i, (crop, bbox, prediction) in enumerate( zip( mole_crops_processed, bboxes, predictions ) ):
            image_filename = f"{request.user_id}_{DETECTION_TYPE}_{request.timestamp}_{i}.jpg"
            crop_path = f"{request.user_id}/{date_str}/{PHOTOS_FOLDER}/{image_filename}"

            # Save image to S3
            try:
                buffer = BytesIO()
                Image.fromarray( crop ).save( buffer, format="JPEG" )
                buffer.seek( 0 )
                s3.get_client().put_object( Bucket=bucket, Key=crop_path, Body=buffer, ContentType="image/jpeg" )
                logger.info( f"Saved crop image to S3: {crop_path}" )
            except NoCredentialsError:
                logger.error( "AWS credentials not found when saving crop image" )
                raise HTTPException( status_code=401, detail="AWS credentials not configured" )
            except ClientError as e:
                logger.error( f"S3 ClientError when saving crop image: {e}" )
                raise HTTPException( status_code=500, detail="Failed to save crop image to S3" )

            # Format result entry
            x, y, w, h = bbox
            label_probs = {CLASS_LABELS[j]: float( prob ) for j, prob in enumerate( prediction )}
            result_entry = {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "label_probabilities": label_probs,
                "image_url": f"s3://{bucket}/{crop_path}"
            }
            all_results.append( result_entry )

        # Save all results as JSON
        results_key = f"{request.user_id}/{date_str}/{RESULTS_FOLDER}/{request.user_id}_{DETECTION_TYPE}_{request.timestamp}.json"
        try:
            s3.get_client().put_object(
                Bucket=bucket,
                Key=results_key,
                Body=json.dumps( {"results": all_results}, ensure_ascii=False ),
                ContentType="application/json"
            )
            logger.info( f"Saved results JSON to S3: {results_key}" )
        except NoCredentialsError:
            logger.error( "AWS credentials not found when saving results JSON" )
            raise HTTPException( status_code=401, detail="AWS credentials not configured" )
        except ClientError as e:
            logger.error( f"S3 ClientError when saving results JSON: {e}" )
            raise HTTPException( status_code=500, detail="Failed to save results JSON to S3" )

        return {
            "path": f"s3://{bucket}/{results_key}",
            "results": all_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception( "Classification failed" )
        raise HTTPException( status_code=500, detail=str( e ) )


@router.post("/classify-mole")
async def classify(request: ClassifyRequest):
    """
        Detect, crop, and classify moles in an image using segmentation and classification models.
        Save cropped images and results back to S3.
        """
    s3 = S3Client()
    try:
        # Parse and validate timestamp
        try:
            timestamp = datetime.fromisoformat(request.timestamp )
        except ValueError:
            raise HTTPException( status_code=400, detail="Invalid timestamp format, expected ISO 8601" )

        date_str = timestamp.strftime( "%Y-%m-%d" )

        # Fetch image from S3
        try:
            image, bucket = s3.fetch_image( request.url )
        except HTTPException:
            # propagate HTTPExceptions from fetch (400, 401, 404)
            raise
        except Exception as e:
            logger.error( f"Failed to fetch image for classification: {e}" )
            raise HTTPException( status_code=500, detail="Failed to download image" )

        # Segmentation stage
        segmented_image, masks = segmentation_pipeline.segment( image )
        mole_crops, bboxes, mask_crops = segmentation_pipeline.extract_moles( segmented_image, masks )
        mole_crops_processed = segmentation_pipeline.extract_moles_from_contours( mole_crops, mask_crops )

        # Classification stage
        predictions = classification_pipeline.predict( mole_crops_processed )

        all_results = []

        for i, (crop, bbox, prediction) in enumerate( zip( mole_crops_processed, bboxes, predictions ) ):
            image_filename = f"{request.user_id}_{DETECTION_TYPE}_{request.timestamp}_{i}.jpg"
            crop_path = f"{request.user_id}/{date_str}/{PHOTOS_FOLDER}/{image_filename}"

            # Save image to S3
            try:
                buffer = BytesIO()
                Image.fromarray( crop ).save( buffer, format="JPEG" )
                buffer.seek( 0 )
                s3.get_client().put_object( Bucket=bucket, Key=crop_path, Body=buffer, ContentType="image/jpeg" )
                logger.info( f"Saved crop image to S3: {crop_path}" )
            except NoCredentialsError:
                logger.error( "AWS credentials not found when saving crop image" )
                raise HTTPException( status_code=401, detail="AWS credentials not configured" )
            except ClientError as e:
                logger.error( f"S3 ClientError when saving crop image: {e}" )
                raise HTTPException( status_code=500, detail="Failed to save crop image to S3" )

            # Format result entry
            x, y, w, h = bbox
            label_probs = {CLASS_LABELS[j]: float( prob ) for j, prob in enumerate( prediction )}
            result_entry = {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "label_probabilities": label_probs,
                "image_url": f"s3://{bucket}/{crop_path}"
            }
            all_results.append( result_entry )

        one_mole = all_results[0]
        one_mole_prob = one_mole["label_probabilities"]
        image_url = one_mole["image_url"]

        # Save all results as JSON
        results_key = f"{request.user_id}/{date_str}/{RESULTS_FOLDER}/{request.user_id}_{DETECTION_TYPE}_{request.timestamp}.json"
        try:
            s3.get_client().put_object(
                Bucket=bucket,
                Key=results_key,
                Body=json.dumps( {**one_mole_prob, "image_url": image_url} )                ,
                ContentType="application/json"
            )
            logger.info( f"Saved results JSON to S3: {results_key}" )
        except NoCredentialsError:
            logger.error( "AWS credentials not found when saving results JSON" )
            raise HTTPException( status_code=401, detail="AWS credentials not configured" )
        except ClientError as e:
            logger.error( f"S3 ClientError when saving results JSON: {e}" )
            raise HTTPException( status_code=500, detail="Failed to save results JSON to S3" )

        return {**one_mole_prob, "results_path": results_key, "image_url": image_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception( "Classification failed" )
        raise HTTPException( status_code=500, detail=str( e ) )

