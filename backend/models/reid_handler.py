import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from backend.config import Config

class ReIDHandler:
    def __init__(self):
        self.model_path = Config.REID_MODEL_PATH
        self.input_size = Config.REID_INPUT_SIZE
        self.session = None
        self.load_model()
    
    def load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path)
            print("Re-ID model loaded successfully")
        except Exception as e:
            print(f"Error loading Re-ID model: {e}")
            self.session = None
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image = image.resize(self.input_size)
        
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def extract_embedding(self, image):
        if self.session is None:
            print("Re-ID model not loaded, returning dummy embedding")
            return np.random.rand(512).tolist()
        
        try:
            input_array = self.preprocess_image(image)
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            embedding = self.session.run([output_name], {input_name: input_array})[0]
            
            embedding = embedding.flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
        
        except Exception as e:
            print(f"Error extracting Re-ID embedding: {e}")
            return np.random.rand(512).tolist()
