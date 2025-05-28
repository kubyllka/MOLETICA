# utils/s3.py

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import HTTPException
from utils.logger import logger

class S3Client:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = boto3.client("s3")
        return cls._client

    @classmethod
    def fetch_image(cls, s3_url: str):
        parsed = urlparse(s3_url)
        if parsed.scheme != "s3":
            raise HTTPException(status_code=400, detail="URL must start with s3://")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            client = cls.get_client()
            response = client.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
            logger.info(f"Successfully fetched image from S3: {bucket}/{key}")
            return Image.open(BytesIO(file_content)).convert("RGB"), bucket
        except Exception as e:
            logger.error(f"Failed to fetch image from S3: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch image from S3")
