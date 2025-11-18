# backend/admin_app.py
"""
Admin backend for MissingPersonDetect
Run: python backend/admin_app.py
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Try operations import
try:
    import backend.database.operations as ops
    from backend.database.operations import (
        add_person, get_all_persons, get_person_by_id,
        update_person_status, add_alert
    )
except Exception:
    ops = None
    add_person = None
    get_all_persons = None
    get_person_by_id = None
    update_person_status = None
    add_alert = None

# UPLOAD FOLDER
try:
    from backend.config import Config
    UPLOAD_FOLDER = Config.UPLOAD_FOLDER
except:
    UPLOAD_FOLDER = os.path.abspath("backend/static/uploads")

# Ensure fallback directories exist
FALLBACK_DIR = os.path.abspath("backend/admin_data")
os.makedirs(FALLBACK_DIR, exist_ok=True)
F_PERSONS = os.path.join(FALLBACK_DIR, "persons.json")
F_DETECTIONS = os.path.join(FALLBACK_DIR, "detections.json")
F_ALERTS = os.path.join(FALLBACK_DIR, "alerts.json")

for f in (F_PERSONS, F_DETECTIONS, F_ALERTS):
    if not os.path.exists(f):
        with open(f, "w") as fh:
            json.dump([], fh)

# === FIXED PATH (Relative & portable) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.join(BASE_DIR, "..", "frontend", "missing person detection")
ADMIN_HTML = "admin.html"

# Flask
app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Utility JSON functions
# --------------------------------------------------
def read_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except:
        return []


def write_json(path, data):
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def next_id(items):
    return max([i.get("id", 0) for i in items] + [0]) + 1


# --------------------------------------------------
#  FIXED STATIC FILE SERVER (CSS/JS/IMAGE WORKS NOW)
# --------------------------------------------------
def find_file_recursive(filename):
    """Search frontend folder recursively for any file."""
    for root, dirs, files in os.walk(FRONTEND_FOLDER):
        if filename in files:
            return os.path.join(root, filename)
    return None


@app.route("/admin/<path:filename>")
def serve_admin_static(filename):
    """
    Handles admin.css, JS files, images
    Searches entire frontend folder recursively
    """
    # Direct path check
    fullpath = os.path.join(FRONTEND_FOLDER, filename)
    if os.path.exists(fullpath):
        return send_from_directory(os.path.dirname(fullpath), os.path.basename(fullpath))

    # Recursive search
    recursive = find_file_recursive(os.path.basename(filename))
    if recursive:
        return send_from_directory(os.path.dirname(recursive), os.path.basename(recursive))

    return jsonify({"error": "file_not_found", "file": filename}), 404


# --------------------------------------------------
# Pages
# --------------------------------------------------
@app.route("/admin")
def serve_admin_home():
    return send_from_directory(FRONTEND_FOLDER, ADMIN_HTML)


@app.route("/admin/uploads/<path:filename>")
def serve_uploaded(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "upload_not_found"}), 404


# --------------------------------------------------
# PERSON LIST
# --------------------------------------------------
@app.route("/admin/api/persons", methods=["GET"])
def admin_persons():
    try:
        if callable(get_all_persons):
            persons = get_all_persons()
            serial = []
            for p in persons:
                serial.append({
                    "id": p.id,
                    "name": p.name,
                    "phone": p.phone,
                    "date_missing": p.date_missing,
                    "attributes": p.attributes,
                    "detected_attributes": p.detected_attributes,
                    "status": p.status,
                    "similarity_score": p.similarity_score,
                    "image_path": p.image_path
                })
            return jsonify({"success": True, "persons": serial})
        else:
            data = read_json(F_PERSONS)
            return jsonify({"success": True, "persons": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------------------------------
# FETCH SINGLE PERSON
# --------------------------------------------------
@app.route("/admin/api/persons/<int:pid>", methods=["GET"])
def admin_get_person(pid):
    try:
        if callable(get_person_by_id):
            p = get_person_by_id(pid)
            if not p:
                return jsonify({"error": "not_found"}), 404
            person = {
                "id": p.id,
                "name": p.name,
                "phone": p.phone,
                "date_missing": p.date_missing,
                "attributes": p.attributes,
                "detected_attributes": p.detected_attributes,
                "status": p.status,
                "similarity_score": p.similarity_score,
                "image_path": p.image_path
            }
            return jsonify({"success": True, "person": person})
        else:
            data = read_json(F_PERSONS)
            for p in data:
                if p["id"] == pid:
                    return jsonify({"success": True, "person": p})
            return jsonify({"error": "not_found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# CREATE PERSON
# --------------------------------------------------
@app.route("/admin/api/persons", methods=["POST"])
def admin_create_person():
    try:
        data = request.form.to_dict()
        name = data.get("name")
        phone = data.get("phone")
        date_missing = data.get("date")

        if not all([name, phone, date_missing]):
            return jsonify({"error": "missing_fields"}), 400

        # File upload
        image_file = request.files.get("image_file")
        filename = None
        if image_file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filename = f"{uuid.uuid4().hex}_{image_file.filename}"
            image_file.save(os.path.join(UPLOAD_FOLDER, filename))

        # If DB exists
        if callable(add_person):
            person_id = add_person(
                name=name,
                phone=phone,
                date_missing=date_missing,
                attributes="{}",
                image_path=os.path.join(UPLOAD_FOLDER, filename) if filename else None,
                reid_emb=None,
                face_emb=None,
                detected_attrs="{}"
            )
            return jsonify({"success": True, "person_id": person_id})

        # fallback JSON
        persons = read_json(F_PERSONS)
        new_id = next_id(persons)
        persons.append({
            "id": new_id,
            "name": name,
            "phone": phone,
            "date_missing": date_missing,
            "status": "missing",
            "image_path": filename
        })
        write_json(F_PERSONS, persons)
        return jsonify({"success": True, "person_id": new_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# DELETE PERSON
# --------------------------------------------------
@app.route("/admin/api/persons/<int:pid>", methods=["DELETE"])
def admin_delete_person(pid):
    try:
        if hasattr(ops, "delete_person"):
            ops.delete_person(pid)
            return jsonify({"success": True})

        persons = read_json(F_PERSONS)
        newp = [p for p in persons if p["id"] != pid]
        write_json(F_PERSONS, newp)
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# 404 FALLBACK
# --------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(FRONTEND_FOLDER, ADMIN_HTML), 200
# UPDATE PERSON STATUS
@app.route("/admin/api/persons/<int:pid>/status", methods=["PUT"])
def update_status(pid):
    try:
        data = request.get_json()
        status = data.get("status")
        if not status:
            return jsonify({"error": "missing status"}), 400

        if callable(update_person_status):
            success = update_person_status(pid, status)
            return jsonify({"success": success})
        return jsonify({"error": "DB function not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    print("Admin UI:", "http://127.0.0.1:5001/admin")
    print("Admin Folder:", FRONTEND_FOLDER)
    app.run(host="0.0.0.0", port=5001, debug=True)
