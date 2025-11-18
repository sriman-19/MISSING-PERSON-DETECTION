import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = "AIzaSyBFYWsb4aZHz0x3ZiaimFlVKQ0vei8nk3I"
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    REID_MODEL_PATH = 'backend/models/reid_model.onnx'
    
    REID_INPUT_SIZE = (256, 128)
    FACE_INPUT_SIZE = (112, 112)
    
    SIMILARITY_THRESHOLD = 0.7
    
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SMTP_EMAIL = os.getenv('SMTP_EMAIL', 'your-email@gmail.com')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    
    POLICE_EMAIL = os.getenv('POLICE_EMAIL', 'police@example.com')
    
    UPLOAD_FOLDER = 'backend/static/uploads'
    MAX_UPLOAD_SIZE = 16 * 1024 * 1024
