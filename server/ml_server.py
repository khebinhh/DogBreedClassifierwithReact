import uvicorn
import torch
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ml_service import classifier, CustomResNet, BREED_CLASSES

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify_image(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        print(f"Processing file: {file.filename}, type: {file.content_type}")
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        print(f"File size: {len(contents)} bytes")
        result = await classifier.classify_image(contents)
        print(f"Classification successful: {result['predictions'][0]}")
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing ML service...")
        # Initialize model at startup
        await classifier.load_model()
        print("ML service initialized successfully")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    try:
        print("Starting FastAPI server on port 8000...")
        uvicorn.run(
            "ml_server:app",
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            access_log=True,
            workers=1
        )
    except Exception as e:
        print(f"Failed to start ML service: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)