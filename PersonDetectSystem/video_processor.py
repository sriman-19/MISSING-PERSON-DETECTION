import cv2
import os
from typing import List, Tuple, Optional
import numpy as np
from face_recognition import extract_faces_from_frame, is_match
from database import get_pending_cases, create_match, update_case_status

def extract_frames(video_path: str, fps: int = 5) -> List[Tuple[np.ndarray, float]]:
    frames_with_timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames_with_timestamps
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30
    
    frame_interval = max(1, int(video_fps / fps))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            frames_with_timestamps.append((frame, timestamp))
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames_with_timestamps)} frames from video")
    return frames_with_timestamps

def process_video(video_path: str, video_location: str = "Unknown Location") -> dict:
    print(f"Processing video: {video_path}")
    
    pending_cases = get_pending_cases()
    print(f"Found {len(pending_cases)} pending cases to match against")
    
    for case in pending_cases:
        update_case_status(case['case_id'], "PROCESSING")
    
    frames_with_timestamps = extract_frames(video_path, fps=5)
    
    if not frames_with_timestamps:
        for case in pending_cases:
            update_case_status(case['case_id'], "QUEUED")
        return {
            "status": "error",
            "message": "Could not extract frames from video"
        }
    
    for case in pending_cases:
        update_case_status(case['case_id'], "MATCHING")
    
    matches_found = {}
    total_faces_detected = 0
    
    for frame_idx, (frame, timestamp) in enumerate(frames_with_timestamps):
        faces = extract_faces_from_frame(frame)
        total_faces_detected += len(faces)
        
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{len(frames_with_timestamps)}, detected {len(faces)} faces")
        
        for face_embedding, bbox in faces:
            for case in pending_cases:
                if case['case_id'] in matches_found:
                    continue
                
                case_embedding = case['embedding']
                match, similarity = is_match(face_embedding, case_embedding)
                
                if match:
                    print(f"MATCH FOUND! Case {case['case_id']}, Similarity: {similarity:.3f}")
                    
                    case_id = case['case_id']
                    results_dir = f"admin_results/{case_id}/found_frames"
                    os.makedirs(results_dir, exist_ok=True)
                    
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Match: {similarity:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    frame_filename = f"match_{int(timestamp)}.jpg"
                    frame_path = os.path.join(results_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    timestamp_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
                    create_match(case_id, frame_path, timestamp_str, similarity)
                    
                    update_case_status(case_id, "PERSON FOUND", video_location)
                    
                    matches_found[case_id] = {
                        "case_id": case_id,
                        "name": case['name'],
                        "frame_path": frame_path,
                        "timestamp": timestamp_str,
                        "similarity": similarity,
                        "location": video_location
                    }
    
    for case in pending_cases:
        if case['case_id'] not in matches_found:
            update_case_status(case['case_id'], "QUEUED")
    
    return {
        "status": "success",
        "frames_processed": len(frames_with_timestamps),
        "faces_detected": total_faces_detected,
        "matches_found": len(matches_found),
        "matches": list(matches_found.values())
    }
