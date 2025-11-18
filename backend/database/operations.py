from backend.database.schema import Session, Person, Detection, Alert
from sqlalchemy import text
from datetime import datetime
import numpy as np

def init_pgvector():
    session = Session()
    try:
        session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        session.commit()
    except Exception as e:
        print(f"pgvector extension creation: {e}")
        session.rollback()
    finally:
        session.close()

def add_person(name, phone, date_missing, attributes, image_path, reid_emb=None, face_emb=None, detected_attrs=''):
    session = Session()
    try:
        person = Person(
            name=name,
            phone=phone,
            date_missing=date_missing,
            attributes=attributes,
            image_path=image_path,
            reid_embedding=reid_emb,
            face_embedding=face_emb,
            detected_attributes=detected_attrs
        )
        session.add(person)
        session.commit()
        person_id = person.id
        return person_id
    except Exception as e:
        session.rollback()
        print(f"Error adding person: {e}")
        return None
    finally:
        session.close()

def get_all_persons():
    session = Session()
    try:
        persons = session.query(Person).all()
        return persons
    finally:
        session.close()

def get_person_by_id(person_id):
    session = Session()
    try:
        person = session.query(Person).filter(Person.id == person_id).first()
        return person
    finally:
        session.close()

def search_similar_persons(reid_emb=None, face_emb=None, top_k=5):
    session = Session()
    try:
        results = []
        
        if reid_emb is not None:
            query = session.query(Person).filter(Person.reid_embedding.isnot(None))
            persons = query.all()
            
            for person in persons:
                if person.reid_embedding:
                    stored_emb = np.array(person.reid_embedding)
                    query_emb = np.array(reid_emb)
                    similarity = np.dot(query_emb, stored_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-8
                    )
                    results.append({
                        'person': person,
                        'similarity': float(similarity),
                        'type': 'reid'
                    })
        
        if face_emb is not None:
            query = session.query(Person).filter(Person.face_embedding.isnot(None))
            persons = query.all()
            
            for person in persons:
                if person.face_embedding:
                    stored_emb = np.array(person.face_embedding)
                    query_emb = np.array(face_emb)
                    similarity = np.dot(query_emb, stored_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-8
                    )
                    
                    existing = next((r for r in results if r['person'].id == person.id), None)
                    if existing:
                        existing['similarity'] = max(existing['similarity'], float(similarity))
                    else:
                        results.append({
                            'person': person,
                            'similarity': float(similarity),
                            'type': 'face'
                        })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    finally:
        session.close()

def add_detection(camera_id, bbox, confidence, image_path, reid_emb=None, face_emb=None, attributes=''):
    session = Session()
    try:
        detection = Detection(
            camera_id=camera_id,
            bbox=str(bbox),
            confidence=confidence,
            image_path=image_path,
            reid_embedding=reid_emb,
            face_embedding=face_emb,
            attributes=attributes
        )
        session.add(detection)
        session.commit()
        return detection.id
    except Exception as e:
        session.rollback()
        print(f"Error adding detection: {e}")
        return None
    finally:
        session.close()

def update_detection_match(detection_id, person_id, similarity):
    session = Session()
    try:
        detection = session.query(Detection).filter(Detection.id == detection_id).first()
        if detection:
            detection.match_found = 1
            detection.matched_person_id = person_id
            detection.similarity_score = similarity
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error updating detection: {e}")
        return False
    finally:
        session.close()

def add_alert(person_id, detection_id, camera_location, recipient):
    session = Session()
    try:
        alert = Alert(
            person_id=person_id,
            detection_id=detection_id,
            camera_location=camera_location,
            recipient=recipient
        )
        session.add(alert)
        session.commit()
        return alert.id
    except Exception as e:
        session.rollback()
        print(f"Error adding alert: {e}")
        return None
    finally:
        session.close()

def update_person_status(person_id, status, similarity_score=None):
    session = Session()
    try:
        person = session.query(Person).filter(Person.id == person_id).first()
        if person:
            person.status = status
            if similarity_score is not None:
                person.similarity_score = similarity_score
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error updating person status: {e}")
        return False
    finally:
        session.close()
