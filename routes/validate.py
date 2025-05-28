# routes/validate.py
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pipelines.autoencoder_pipeline import AutoencoderPipeline, load_autoencoder
from utils.s3 import S3Client
from utils.logger import logger

router = APIRouter()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_state = 4747

validation_model = load_autoencoder("models/autoencoder.pth", device)
autoencoder_pipeline = AutoencoderPipeline(validation_model, device)

class ValidateRequest(BaseModel):
    url: str = Field(..., description="S3 URL")

@router.post("/validate-mole")
async def validate(request: ValidateRequest):
    """
        Validate if the image likely contains a mole using an autoencoder model.
        """
    try:
        image, _ = S3Client.fetch_image( request.url )
        result = autoencoder_pipeline.validate( image )

        is_mole = result['PSNR'] >= 24
        logger.info( f"Validation PSNR: {result['PSNR']} - Is mole: {is_mole}" )
        return {"value": bool( is_mole )}
    except Exception as e:
        logger.exception( "Validation failed" )
        raise HTTPException( status_code=500, detail=str( e ) )

