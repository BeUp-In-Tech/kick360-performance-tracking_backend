from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import cv2
import shutil
import os
import numpy as np
import time
import gc
from ultralytics import YOLO
import supervision as sv
from engine import SoccerAnalytics
from contextlib import asynccontextmanager

# =========================
# CONFIG
# =========================
PROCESS_SECONDS = 5      # only process first 5 seconds
MAX_FRAMES = 5           # max frames to process
MAX_PROCESS_TIME = 6     # safety timeout

# =========================
# MODEL LOAD
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading YOLO model...")
    app.state.model = YOLO("yolov8n.pt")
    print("✅ Model loaded!")
    yield

app = FastAPI(
    title="Kick 360 Backend (Smart Processing)",
    version="4.0.0",
    lifespan=lifespan
)

tracker = sv.ByteTrack()
analytics = SoccerAnalytics()

# =========================
# HEALTH
# =========================
@app.get("/")
async def health_check():
    return {"status": "online", "message": "Smart backend running"}

# =========================
# CLEANUP
# =========================
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# =========================
# ANALYZE
# =========================
@app.post("/analyze/")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    model = app.state.model

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration_seconds = frame_count / fps if fps > 0 else 0

    final_output = {}

    # =========================
    # 🎯 PROCESS ONLY FIRST N SECONDS
    # =========================
    max_frames_to_read = int(fps * PROCESS_SECONDS)

    frame_interval = max(1, int(max_frames_to_read / MAX_FRAMES))

    frame_index = 0
    processed_frames = 0

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # stop after N seconds
        if frame_index >= max_frames_to_read:
            break

        # time safety
        if time.time() - start_time > MAX_PROCESS_TIME:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        frame_index += 1
        processed_frames += 1

        # 🔻 downscale
        frame = cv2.resize(frame, (192, 108))

        # 🔻 light inference
        results = model(frame, verbose=False, conf=0.6, imgsz=192)[0]
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

        if processed_frames >= MAX_FRAMES:
            break

    cap.release()
    background_tasks.add_task(remove_file, file_path)

    gc.collect()

    return {
        "status": "success",
        "video_duration": round(duration_seconds, 2),
        "processed_seconds": PROCESS_SECONDS,
        "players_detected": len(final_output)
    }