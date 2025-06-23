import os
import shutil
import glob
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import the generate function from app.py
from app_acc import generate


def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/generate")
async def generate_api(
    image_input: UploadFile = File(...),
    audio_input: UploadFile = File(...),
    pose_input: str = Form(...),
    width: int = Form(768),
    height: int = Form(768),
    length: int = Form(120),
    steps: int = Form(30),
    sample_rate: int = Form(16000),
    cfg: float = Form(2.5),
    fps: int = Form(24),
    context_frames: int = Form(12),
    context_overlap: int = Form(3),
    quantization_input: bool = Form(False),
    seed: int = Form(-1),
):
    image_path = f"temp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_image.png"
    audio_path = f"temp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_audio.wav"
    os.makedirs("temp", exist_ok=True)
    save_upload_file(image_input, image_path)
    save_upload_file(audio_input, audio_path)
    try:
        video_output, _ = generate(
            image_path,
            audio_path,
            pose_input,
            width,
            height,
            length,
            steps,
            sample_rate,
            cfg,
            fps,
            context_frames,
            context_overlap,
            quantization_input,
            seed,
        )
        return FileResponse(
            video_output,
            media_type="video/mp4",
            filename=os.path.basename(video_output),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/outputs")
async def list_outputs_api():
    files = sorted(glob.glob("outputs/*.mp4"), reverse=True)
    return {"files": [os.path.basename(f) for f in files]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
