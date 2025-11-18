import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
from backend.config import Config

class FaceHandler:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.input_size = Config.FACE_INPUT_SIZE
    
    def detect_face(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if not results.detections:
            return None
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = image.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        face_crop = image[y:y+height, x:x+width]
        
        return face_crop
    
    def extract_face_embedding(self, image):
        try:
            if isinstance(image, str):
                image = cv2.imread(image)
            elif isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            face_crop = self.detect_face(image)
            
            if face_crop is None or face_crop.size == 0:
                print("No face detected, returning dummy embedding")
                return np.random.rand(128).tolist()
            
            face_resized = cv2.resize(face_crop, self.input_size)
            
            rgb_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)
            
            if not results.multi_face_landmarks:
                print("No face landmarks detected, returning dummy embedding")
                return np.random.rand(128).tolist()
            
            landmarks = results.multi_face_landmarks[0]
            
            embedding = []
            for landmark in landmarks.landmark[:128]:
                embedding.extend([landmark.x, landmark.y, landmark.z])
            
            embedding = embedding[:128]
            while len(embedding) < 128:
                embedding.append(0.0)
            
            embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
        
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return np.random.rand(128).tolist()
