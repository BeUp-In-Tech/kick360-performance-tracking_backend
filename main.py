from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import cv2
import shutil
import os
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv
from engine import SoccerAnalytics
from contextlib import asynccontextmanager

# =========================
# CONFIG (VERY IMPORTANT)
# =========================
MAX_VIDEO_DURATION = 6   # seconds
MAX_PROCESS_TIME = 8     # seconds
MAX_FRAMES = 12          # total frames to process

# =========================
# LOAD MODEL (SAFE)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading YOLO model...")
    app.state.model = YOLO("yolov8n.pt")
    print("✅ Model loaded!")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Kick 360 Backend (Render Optimized)",
    version="2.0.0",
    lifespan=lifespan
)

tracker = sv.ByteTrack()
analytics = SoccerAnalytics()

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "message": "Kick 360 backend running (optimized)",
        "docs": "/docs"
    }

# =========================
# CLEANUP
# =========================
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# =========================
# MAIN ANALYSIS
# =========================
@app.post("/analyze/")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    model = app.state.model

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration_seconds = round(frame_count / fps, 2) if fps > 0 else 0

    # =========================
    # 🚨 LIMIT VIDEO LENGTH
    # =========================
    if duration_seconds > MAX_VIDEO_DURATION:
        cap.release()
        background_tasks.add_task(remove_file, file_path)
        return {
            "status": "error",
            "message": f"Video too long. Max allowed is {MAX_VIDEO_DURATION} seconds."
        }

    final_output = {}

    # =========================
    # FRAME SAMPLING (IMPORTANT)
    # =========================
    frame_interval = max(1, int(frame_count / MAX_FRAMES))

    frame_index = 0
    processed_frames = 0

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ⏱ TIME LIMIT (prevents 502)
        if time.time() - start_time > MAX_PROCESS_TIME:
            print("⏹ Time limit reached, stopping early")
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        frame_index += 1
        processed_frames += 1

        # =========================
        # 🔥 REDUCE VIDEO QUALITY
        # =========================
        frame = cv2.resize(frame, (256, 144))

        # =========================
        # LIGHT YOLO
        # =========================
        results = model(frame, verbose=False, conf=0.5, imgsz=256)[0]
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

    # =========================
    # RESPONSE (LIGHTWEIGHT)
    # =========================
    return {
        "status": "success",
        "video_info": {
            "duration_sec": duration_seconds,
            "processed_frames": processed_frames
        },
        "players_detected": len(final_output),
        "analysis": final_output
    }