import mediapipe as mp
import numpy as np
import cv2
from typing import Optional, List, Tuple
import os

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FACE_DETECTOR_MODEL = "detector.tflite"
SIMILARITY_THRESHOLD = 0.75

def download_mediapipe_models():
    import urllib.request
    
    detector_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    if not os.path.exists(FACE_DETECTOR_MODEL):
        print("Downloading MediaPipe face detector model...")
        urllib.request.urlretrieve(detector_url, FACE_DETECTOR_MODEL)
        print("Model downloaded successfully!")

def extract_face_embedding(image_path: str) -> Optional[np.ndarray]:
    try:
        download_mediapipe_models()
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        
        with FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            
            if not detection_result.detections:
                print(f"No face detected in {image_path}")
                return None
            
            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            
            face_roi = rgb_image[y:y2, x:x2]
            
            if face_roi.size == 0:
                print(f"Invalid face ROI in {image_path}")
                return None
            
            face_resized = cv2.resize(face_roi, (128, 128))
            embedding = face_resized.flatten().astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            
            return embedding
            
    except Exception as e:
        print(f"Error extracting face embedding from {image_path}: {str(e)}")
        return None

def extract_faces_from_frame(frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    try:
        download_mediapipe_models()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        
        faces = []
        
        with FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                
                face_roi = rgb_frame[y:y2, x:x2]
                
                if face_roi.size == 0:
                    continue
                
                face_resized = cv2.resize(face_roi, (128, 128))
                embedding = face_resized.flatten().astype(np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                
                faces.append((embedding, (x, y, x2, y2)))
        
        return faces
        
    except Exception as e:
        print(f"Error extracting faces from frame: {str(e)}")
        return []

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def is_match(embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity >= SIMILARITY_THRESHOLD, similarity
