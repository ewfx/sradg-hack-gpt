import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import uvicorn
from models.anomaly_detection import detect_anomalies
from models.llm_fix_suggestion import generate_fix_suggestion
from backend.fix_executor import execute_fix
from pydantic import BaseModel
from typing import Dict, Any
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class AnomalyRequest(BaseModel):
    key_columns: list
    break_category: list
    record: Dict[str, Any]

@app.get("/")
def home():
    return {"message": "Welcome to AI-Powered Reconciliation!"}

@app.post("/detect_anomaly/")
def detect_anomaly(request: AnomalyRequest):
    logger.info(f"Received anomaly detection request: {request.record}")
    anomaly_result = detect_anomalies(
        request.record,
        request.break_category,
        request.key_columns
    )
    fix_suggestion = generate_fix_suggestion(
        request.key_columns,
        request.break_category,
        request.record,
        anomaly_result
    )
    logger.info(f"Anomaly result: {anomaly_result}, Fix suggestion: {fix_suggestion}")
    return {
        "anomaly_result": anomaly_result,
        "fix_suggestion": fix_suggestion
    }

class FixApproval(BaseModel):
    record: Dict[str, Any]
    approval: bool
    feedback: str

@app.post("/approve_fix/")
def approve_fix(request: FixApproval):
    logger.info(f"Received fix approval request: {request}")
    if request.approval:
        fix_result = execute_fix(request.record)
        logger.info(f"Fix executed successfully: {fix_result}")
        return {"status": "Fix executed", "result": fix_result}
    else:
        logger.info(f"Fix rejected with feedback: {request.feedback}")
        return {"status": "Fix rejected", "feedback": request.feedback}

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST"), port=os.getenv("API_PORT"))