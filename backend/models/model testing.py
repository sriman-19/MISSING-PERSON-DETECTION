import onnxruntime

model_path = "C:/Users/Sriman Reddy/Downloads/MissingPersonDetect/MissingPersonDetect/backend/models/reid_model.onnx"
session = onnxruntime.InferenceSession(model_path)
print("Model loaded successfully!")
