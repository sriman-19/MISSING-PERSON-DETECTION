from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
import base64

from backend.config import Config
from backend.database.operations import (
    init_pgvector, add_person, get_all_persons, get_person_by_id,
    search_similar_persons, add_detection, update_detection_match,
    add_alert, update_person_status
)
from backend.models.reid_handler import ReIDHandler
from backend.models.face_handler import FaceHandler
from backend.models.gemini_handler import GeminiHandler
from backend.models.yolo_handler import YOLOHandler
from backend.utils.image_utils import save_base64_image, image_to_base64
from backend.utils.alert_utils import send_alert
from backend.utils.camera_utils import CameraSimulator

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE

# initialize DB/vector store
init_pgvector()

# model handlers
reid_handler = ReIDHandler()
face_handler = FaceHandler()
gemini_handler = GeminiHandler()
yolo_handler = YOLOHandler()

camera_simulator = CameraSimulator('CAM-001')


# -----------------------
# API ROUTES (backend)
# -----------------------

@app.route('/api/upload', methods=['POST'])
def upload_person():
    try:
        data = request.json

        name = data.get('name')
        phone = data.get('phone')
        date_missing = data.get('date')
        attributes = data.get('attributes')
        image_base64 = data.get('image')

        if not all([name, phone, date_missing, image_base64]):
            return jsonify({'error': 'Missing required fields'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"person_{name.replace(' ', '_')}_{timestamp}.jpg"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

        if not save_base64_image(image_base64, filepath):
            return jsonify({'error': 'Failed to save image'}), 500

        reid_embedding = reid_handler.extract_embedding(filepath)
        face_embedding = face_handler.extract_face_embedding(filepath)

        detected_attrs = gemini_handler.detect_attributes(filepath)
        detected_attrs_str = json.dumps(detected_attrs)

        person_id = add_person(
            name=name,
            phone=phone,
            date_missing=date_missing,
            attributes=attributes,
            image_path=filepath,
            reid_emb=reid_embedding,
            face_emb=face_embedding,
            detected_attrs=detected_attrs_str
        )

        if person_id:
            return jsonify({
                'success': True,
                'person_id': person_id,
                'message': 'Person details uploaded successfully',
                'detected_attributes': detected_attrs
            }), 200
        else:
            return jsonify({'error': 'Failed to save to database'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_person():
    try:
        data = request.json
        image_base64 = data.get('image')
        attributes_filter = data.get('attributes', {})
        top_k = data.get('top_k', 5)

        if not image_base64:
            return jsonify({'error': 'Image is required'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"search_{timestamp}.jpg"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

        if not save_base64_image(image_base64, filepath):
            return jsonify({'error': 'Failed to save image'}), 500

        reid_embedding = reid_handler.extract_embedding(filepath)
        face_embedding = face_handler.extract_face_embedding(filepath)

        matches = search_similar_persons(reid_embedding, face_embedding, top_k)

        results = []
        for match in matches:
            person = match['person']
            similarity = match['similarity']

            if similarity >= Config.SIMILARITY_THRESHOLD:
                person_image_b64 = None
                if person.image_path and os.path.exists(person.image_path):
                    person_image_b64 = image_to_base64(person.image_path)

                results.append({
                    'person_id': person.id,
                    'name': person.name,
                    'phone': person.phone,
                    'date_missing': person.date_missing,
                    'attributes': person.attributes,
                    'detected_attributes': person.detected_attributes,
                    'similarity': similarity,
                    'status': person.status,
                    'image': person_image_b64
                })

        return jsonify({
            'success': True,
            'matches_found': len(results),
            'matches': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect_from_camera():
    try:
        data = request.json
        camera_id = data.get('camera_id', 'CAM-001')
        frame_base64 = data.get('frame')

        if frame_base64:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"frame_{camera_id}_{timestamp}.jpg"
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            save_base64_image(frame_base64, filepath)
            frame_path = filepath
        else:
            camera_simulator.add_sample_image('backend/static/uploads/person_sample.jpg')
            detection_result = camera_simulator.simulate_detection_pipeline(
                yolo_handler, reid_handler, face_handler, gemini_handler
            )

            detections_info = []
            for idx, det in enumerate(detection_result['detections']):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                crop_filename = f"detection_{camera_id}_{timestamp}_{idx}.jpg"
                crop_path = os.path.join(Config.UPLOAD_FOLDER, crop_filename)

                import cv2
                cv2.imwrite(crop_path, det['crop'])

                detection_id = add_detection(
                    camera_id=camera_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    image_path=crop_path,
                    reid_emb=det['reid_embedding'],
                    face_emb=det['face_embedding'],
                    attributes=json.dumps(det['attributes'])
                )

                matches = search_similar_persons(det['reid_embedding'], det['face_embedding'], top_k=1)

                match_info = None
                if matches and matches[0]['similarity'] >= Config.SIMILARITY_THRESHOLD:
                    match = matches[0]
                    person = match['person']
                    similarity = match['similarity']

                    update_detection_match(detection_id, person.id, similarity)
                    update_person_status(person.id, 'found', similarity)

                    match_info = {
                        'person_id': person.id,
                        'name': person.name,
                        'phone': person.phone,
                        'similarity': similarity,
                        'status': 'Match Found!'
                    }

                detections_info.append({
                    'detection_id': detection_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'attributes': det['attributes'],
                    'match': match_info
                })

            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'detections_count': len(detections_info),
                'detections': detections_info
            }), 200

        # If a frame was provided we run the normal detection flow
        detections = yolo_handler.detect_persons(frame_path)

        results = []
        for idx, det in enumerate(detections):
            person_crop = det['crop']

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            crop_filename = f"detection_{camera_id}_{timestamp}_{idx}.jpg"
            crop_path = os.path.join(Config.UPLOAD_FOLDER, crop_filename)

            import cv2
            cv2.imwrite(crop_path, person_crop)

            reid_embedding = reid_handler.extract_embedding(person_crop)
            face_embedding = face_handler.extract_face_embedding(person_crop)
            attributes = gemini_handler.detect_attributes(person_crop)

            detection_id = add_detection(
                camera_id=camera_id,
                bbox=det['bbox'],
                confidence=det['confidence'],
                image_path=crop_path,
                reid_emb=reid_embedding,
                face_emb=face_embedding,
                attributes=json.dumps(attributes)
            )

            matches = search_similar_persons(reid_embedding, face_embedding, top_k=1)

            match_info = None
            if matches and matches[0]['similarity'] >= Config.SIMILARITY_THRESHOLD:
                match = matches[0]
                person = match['person']
                similarity = match['similarity']

                update_detection_match(detection_id, person.id, similarity)

                match_info = {
                    'person_id': person.id,
                    'name': person.name,
                    'similarity': similarity
                }

            results.append({
                'detection_id': detection_id,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'attributes': attributes,
                'match': match_info
            })

        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'detections_count': len(results),
            'detections': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alert', methods=['POST'])
def send_police_alert():
    try:
        data = request.json
        person_id = data.get('person_id')
        camera_location = data.get('camera_location', 'Unknown Location')
        recipient_email = data.get('recipient_email')

        if not person_id:
            return jsonify({'error': 'person_id is required'}), 400

        person = get_person_by_id(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404

        success = send_alert(
            person_name=person.name,
            person_phone=person.phone,
            camera_location=camera_location,
            image_path=person.image_path,
            similarity_score=person.similarity_score or 0.0,
            recipient_email=recipient_email
        )

        if success:
            alert_id = add_alert(
                person_id=person_id,
                detection_id=None,
                camera_location=camera_location,
                recipient=recipient_email or Config.POLICE_EMAIL
            )

            return jsonify({
                'success': True,
                'alert_id': alert_id,
                'message': 'Alert sent successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Alert logged but email not sent (check SMTP configuration)'
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persons', methods=['GET'])
def get_persons():
    try:
        persons = get_all_persons()

        results = []
        for person in persons:
            person_image_b64 = None
            if person.image_path and os.path.exists(person.image_path):
                person_image_b64 = image_to_base64(person.image_path)

            results.append({
                'id': person.id,
                'name': person.name,
                'phone': person.phone,
                'date_missing': person.date_missing,
                'attributes': person.attributes,
                'detected_attributes': person.detected_attributes,
                'status': person.status,
                'similarity_score': person.similarity_score,
                'image': person_image_b64,
                'timestamp': person.timestamp.isoformat() if person.timestamp else None
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'persons': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persons/<int:person_id>', methods=['GET'])
def get_person(person_id):
    try:
        person = get_person_by_id(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404

        person_image_b64 = None
        if person.image_path and os.path.exists(person.image_path):
            person_image_b64 = image_to_base64(person.image_path)

        return jsonify({
            'success': True,
            'person': {
                'id': person.id,
                'name': person.name,
                'phone': person.phone,
                'date_missing': person.date_missing,
                'attributes': person.attributes,
                'detected_attributes': person.detected_attributes,
                'status': person.status,
                'similarity_score': person.similarity_score,
                'image': person_image_b64,
                'timestamp': person.timestamp.isoformat() if person.timestamp else None
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<int:person_id>', methods=['GET'])
def get_status(person_id):
    try:
        person = get_person_by_id(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404

        return jsonify({
            'success': True,
            'person_id': person.id,
            'name': person.name,
            'status': person.status,
            'similarity_score': person.similarity_score
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------
# FRONTEND (serve static)
# -----------------------

# Build absolute path to:  ../frontend/missing person detection
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'missing person detection')
FRONTEND_FOLDER = os.path.abspath(FRONTEND_FOLDER)

@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(FRONTEND_FOLDER, filename)

# -----------------------
# MAIN
# -----------------------
if __name__ == '__main__':
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
