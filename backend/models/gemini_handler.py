import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from backend.config import Config
import json

class GeminiHandler:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def detect_attributes(self, image):
        try:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            prompt = """Analyze this person image and provide the following attributes in JSON format:
{
    "gender": "male/female/unknown",
    "age_range": "child/teen/young_adult/middle_aged/senior",
    "clothing_upper": "description of upper clothing color and type",
    "clothing_lower": "description of lower clothing color and type",
    "accessories": "any visible accessories like hat, bag, glasses",
    "hair": "hair color and style",
    "build": "slim/average/heavy"
}

Only return the JSON, no additional text."""
            
            response = self.model.generate_content([prompt, image])
            
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            attributes = json.loads(text)
            return attributes
        
        except Exception as e:
            print(f"Error detecting attributes with Gemini: {e}")
            return {
                "gender": "unknown",
                "age_range": "unknown",
                "clothing_upper": "unknown",
                "clothing_lower": "unknown",
                "accessories": "none",
                "hair": "unknown",
                "build": "average"
            }
