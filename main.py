from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from bgapi import process_image
from fastapi.middleware.cors import CORSMiddleware
import io


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def home():
    return {"message": "YOLO + BiRefNet Background Remover API"}

@app.post("/remove-background/")
def remove_background(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    image_bytes = file.file.read()
    try:
        result_bytes = process_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")
