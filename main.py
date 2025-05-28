# main.py
from fastapi import FastAPI
from routes.validate import router as validate_router
from routes.classify import router as classify_router

app = FastAPI(
    title="Skin Lesion API MOLETICA",
    version="1.0.0",
    description="API for mole validation and classification"
)

app.include_router(validate_router)
app.include_router(classify_router)
