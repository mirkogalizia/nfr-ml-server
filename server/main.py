from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import subprocess
import json
import os
import time
from datetime import datetime

app = FastAPI(title="NFR ML Server", version="1.0.0")

class HealthResponse(BaseModel):
    status: str
    tensorflow_version: str
    gpus: list[str]

class CommandResponse(BaseModel):
    status: str
    message: str
    output: Optional[str] = ""
    error: Optional[str] = ""

class TrainResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None

# Storage per status training
training_status = {}

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

# ==================== GIT MANAGEMENT ====================

@app.post("/git/pull", response_model=CommandResponse)
async def git_pull():
    """Esegue git pull per aggiornare il codice da GitHub"""
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd="/home/mirko/nfr-ml",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return CommandResponse(
            status="success" if result.returncode == 0 else "error",
            message="Git pull completed" if result.returncode == 0 else "Git pull failed",
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else ""
        )
    except Exception as e:
        return CommandResponse(
            status="error",
            message="Exception during git pull",
            output="",
            error=str(e)
        )

@app.get("/git/status")
async def git_status():
    """Mostra lo stato del repo Git"""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd="/home/mirko/nfr-ml",
            capture_output=True,
            text=True
        )
        
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd="/home/mirko/nfr-ml",
            capture_output=True,
            text=True
        )
        
        return {
            "status": "ok",
            "branch": branch_result.stdout.strip(),
            "changes": result.stdout,
            "is_clean": len(result.stdout.strip()) == 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==================== FILE UPLOAD ====================

@app.post("/upload/service-account")
async def upload_service_account(file: UploadFile = File(...)):
    """Carica il file serviceAccountKey.json"""
    try:
        file_path = "/home/mirko/nfr-ml/serviceAccountKey.json"
        
        # Leggi contenuto
        content = await file.read()
        
        # Salva file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Verifica che sia un JSON valido
        with open(file_path, "r") as f:
            json.load(f)
        
        return {
            "status": "success",
            "message": "Service account key uploaded successfully",
            "file_path": file_path,
            "file_size": len(content)
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Invalid JSON file"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/upload/service-account/status")
async def check_service_account():
    """Verifica se il service account key esiste"""
    file_path = "/home/mirko/nfr-ml/serviceAccountKey.json"
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return {
            "status": "exists",
            "file_path": file_path,
            "file_size": file_size
        }
    else:
        return {
            "status": "missing",
            "message": "Service account key not found"
        }

# ==================== DATA PREPARATION ====================

@app.post("/train/prepare-data", response_model=TrainResponse)
async def prepare_training_data(background_tasks: BackgroundTasks):
    """Prepara i dati per il training (esegue prepare_training_data.py)"""
    
    task_id = f"prepare_data_{int(time.time())}"
    training_status[task_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    def run_script():
        try:
            result = subprocess.run(
                ["python", "scripts/prepare_training_data.py"],
                cwd="/home/mirko/nfr-ml",
                capture_output=True,
                text=True,
                timeout=600  # 10 minuti max
            )
            
            if result.returncode == 0:
                training_status[task_id] = {
                    "status": "completed",
                    "output": result.stdout,
                    "error": None,
                    "completed_at": datetime.now().isoformat()
                }
            else:
                training_status[task_id] = {
                    "status": "failed",
                    "output": result.stdout,
                    "error": result.stderr,
                    "completed_at": datetime.now().isoformat()
                }
        except subprocess.TimeoutExpired:
            training_status[task_id] = {
                "status": "failed",
                "error": "Script timeout (>10 min)",
                "completed_at": datetime.now().isoformat()
            }
        except Exception as e:
            training_status[task_id] = {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
    
    background_tasks.add_task(run_script)
    
    return TrainResponse(
        status="started",
        message="Data preparation started in background",
        task_id=task_id
    )

@app.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Controlla lo status di un task di training"""
    if task_id not in training_status:
        return {"error": "Task not found"}
    
    return training_status[task_id]

@app.get("/train/status")
async def list_all_tasks():
    """Lista tutti i task di training"""
    return {
        "tasks": training_status,
        "count": len(training_status)
    }

# ==================== DATA INFO ====================

@app.get("/data/info")
async def get_data_info():
    """Restituisce info sui dataset disponibili"""
    
    metadata_path = "/home/mirko/nfr-ml/data/metadata.json"
    
    if not os.path.exists(metadata_path):
        return {"error": "No data prepared yet. Run /train/prepare-data first."}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

# ==================== SYSTEM INFO ====================

@app.get("/system/info")
async def system_info():
    """Info sul sistema"""
    try:
        # Disk usage
        disk = subprocess.run(
            ["df", "-h", "/home/mirko/nfr-ml"],
            capture_output=True,
            text=True
        )
        
        return {
            "disk_usage": disk.stdout,
            "cwd": os.getcwd(),
            "python_version": subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True
            ).stdout.strip()
        }
    except Exception as e:
        return {"error": str(e)}
