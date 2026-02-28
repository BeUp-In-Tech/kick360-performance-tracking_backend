from fastapi import FastAPI, UploadFile, File, status
import cv2
import shutil
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from engine import SoccerAnalytics

app = FastAPI(
    title="Kick 360 Backend",
    description="Professional Soccer Analytics API with Video Duration Tracking",
    version="1.1.0"
)

# মডেল এবং ট্র্যাকার লোড
MODEL_PATH = "yolov8n.pt" 
model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()
analytics = SoccerAnalytics()

@app.post("/analyze/", tags=["Kick 360 Analytics Engine"])
async def analyze_video(file: UploadFile = File(...)):
    
    upload_dir = "uploads"
    if not os.path.exists(upload_dir): os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)
    
    # --- ভিডিওর ডিউরেশন ক্যালকুলেশন ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = round(frame_count / fps, 2) if fps > 0 else 0

    final_output = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # AI ডিটেকশন
        results = model(frame, verbose=False, conf=0.25, imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Class 0: Player ফিল্টারিং
        player_detections = tracker.update_with_detections(detections[detections.class_id == 0])

        if player_detections.tracker_id is not None:
            for i, p_id in enumerate(player_detections.tracker_id):
                x1, y1, x2, y2 = player_detections.xyxy[i]
                feet_pos = np.array([(x1+x2)/2, y2])
                
                # ইঞ্জিন থেকে মেট্রিক নেওয়া (PAC, SHO, PAS, DRI, DEF, PHY)
                stats = analytics.get_stats(p_id, feet_pos)
                final_output[str(p_id)] = stats

    cap.release()
    
    # --- নতুন রেসপন্স ফরম্যাট ---
    return {
        "status": "success",
        "response_code": status.HTTP_200_OK,
        "video_info": {
            "filename": file.filename,
            "total_duration_sec": duration_seconds,
            "fps": round(fps, 2)
        },
        "data": {
            "total_players_detected": len(final_output),
            "analysis": final_output
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)