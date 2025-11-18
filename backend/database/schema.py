import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()  # loads DATABASE_URL from .env
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

# ---------------- BASE & ENGINE ----------------
Base = declarative_base()
engine = create_engine(DATABASE_URL)

# ---------------- TABLES ----------------

class Person(Base):
    __tablename__ = 'persons'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200))
    phone = Column(String(50))
    date_missing = Column(String(50))
    attributes = Column(Text)
    image_path = Column(String(500))
    reid_embedding = Column(ARRAY(Float))       # Changed from Vector(512)
    face_embedding = Column(ARRAY(Float))       # Changed from Vector(128)
    detected_attributes = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String(50))
    status = Column(String(50), default='missing')
    similarity_score = Column(Float, default=0.0)

class Detection(Base):
    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer)
    camera_id = Column(String(50))
    detection_time = Column(DateTime, default=datetime.utcnow)
    bbox = Column(String(200))
    confidence = Column(Float)
    image_path = Column(String(500))
    reid_embedding = Column(ARRAY(Float))      # Changed
    face_embedding = Column(ARRAY(Float))      # Changed
    attributes = Column(Text)
    match_found = Column(Integer, default=0)
    matched_person_id = Column(Integer)
    similarity_score = Column(Float)

class Alert(Base):
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer)
    detection_id = Column(Integer)
    alert_time = Column(DateTime, default=datetime.utcnow)
    camera_location = Column(String(200))
    status = Column(String(50), default='sent')
    recipient = Column(String(200))

# ---------------- CREATE TABLES ----------------
def init_db():
    """
    Creates all tables in the database if they don't exist.
    """
    Base.metadata.create_all(engine)
    print("Tables created successfully!")

# ---------------- SESSION ----------------
Session = sessionmaker(bind=engine)

# ---------------- RUN DIRECTLY ----------------
if __name__ == "__main__":
    init_db()
