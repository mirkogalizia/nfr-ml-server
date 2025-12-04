@app.get("/predict/top-variants")
async def predict_top_variants(top_n: int = 50):
    """
    Genera forecast per i top N variants per volume vendite
    """
    try:
        result = subprocess.run(
            ["python", "models/predict_batch.py", str(top_n)],
            cwd="/home/mirko/nfr-ml",
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            predictions = json.loads(result.stdout)
            return predictions
        else:
            return {
                "status": "error",
                "message": "Batch prediction failed",
                "details": result.stderr
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
