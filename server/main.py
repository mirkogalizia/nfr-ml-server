from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI(title="NFR ML Server", version="1.0.0")

class HealthResponse(BaseModel):
    status: str
    tensorflow_version: str
    gpus: list[str]

@app.get("/health", response_model=HealthResponse)
async def health():
    gpus = tf.config.list_physical_devices("GPU")
    return HealthResponse(
        status="ok",
        tensorflow_version=tf.__version__,
        gpus=[str(g) for g in gpus],
    )

@app.get("/")
async def root():
    return {"message": "NFR ML Server running"}
