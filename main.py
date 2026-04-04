from fastapi import FastAPI, UploadFile, File, status, BackgroundTasks
import cv2
import shutil
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from engine import SoccerAnalytics
from contextlib import asynccontextmanager

# ✅ Lifespan (modern startup handler)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading YOLO model...")
    app.state.model = YOLO("yolov8n.pt")
    print("✅ Model loaded!")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Kick 360 Backend",
    description="Optimized Cloud Version",
    version="1.1.0",
    lifespan=lifespan
)

tracker = sv.ByteTrack()
analytics = SoccerAnalytics()

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "message": "Kick 360 Backend is running!",
        "docs": "/docs"
    }

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@app.post("/analyze/")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    model = app.state.model  # ✅ access model safely

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # Save video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = round(frame_count / fps, 2) if fps > 0 else 0

    # ✅ LIMIT VIDEO LENGTH (prevents timeout)
    if duration_seconds > 15:
        cap.release()
        background_tasks.add_task(remove_file, file_path)
        return {
            "status": "error",
            "message": "Video too long. Max allowed is 15 seconds."
        }

    final_output = {}

    # ✅ HEAVY OPTIMIZATION
    frame_skip = 5   # reduce load
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % frame_skip != 0:
            continue

        # ✅ Resize frame (huge performance boost)
        frame = cv2.resize(frame, (640, 360))

        # ✅ Lighter YOLO inference
        results = model(frame, verbose=False, conf=0.3, imgsz=480)[0]
        detections = sv.Detections.from_ultralytics(results)

        player_detections = tracker.update_with_detections(
            detections[detections.class_id == 0]
        )

        if player_detections.tracker_id is not None:
            for i, p_id in enumerate(player_detections.tracker_id):
                x1, y1, x2, y2 = player_detections.xyxy[i]
                feet_pos = np.array([(x1 + x2) / 2, y2])

                stats = analytics.get_stats(p_id, feet_pos)
                final_output[str(p_id)] = stats

    cap.release()

    # ✅ cleanup
    background_tasks.add_task(remove_file, file_path)

    return {
        "status": "success",
        "video_info": {
            "filename": file.filename,
            "duration_sec": duration_seconds,
            "fps": round(fps, 2)
        },
        "players_detected": len(final_output),
        "analysis": final_output
    }