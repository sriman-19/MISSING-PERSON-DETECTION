from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from typing import Optional
import uvicorn
import secrets
import hashlib

from database import (
    init_database, create_case, get_case, update_case_embedding,
    update_case_status, get_matches_for_case, get_all_cases
)
from face_recognition import extract_face_embedding
from video_processor import process_video

app = FastAPI(title="Missing Person Detection System")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = hashlib.sha256(os.getenv("ADMIN_PASSWORD", "admin123").encode()).hexdigest()
active_sessions = {}

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/admin_results", StaticFiles(directory="admin_results"), name="admin_results")

templates = Jinja2Templates(directory="templates")

init_database()

os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/videos", exist_ok=True)
os.makedirs("admin_results", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def verify_admin_session(session_id: Optional[str]) -> bool:
    if not session_id:
        return False
    return session_id in active_sessions

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/api/admin/login")
async def admin_login(
    username: str = Form(...),
    password: str = Form(...)
):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if username == ADMIN_USERNAME and password_hash == ADMIN_PASSWORD_HASH:
        session_id = secrets.token_urlsafe(32)
        active_sessions[session_id] = {"username": username}
        
        json_response = JSONResponse({"success": True})
        json_response.set_cookie(key="admin_session", value=session_id, httponly=True, max_age=86400)
        return json_response
    else:
        return JSONResponse({"success": False, "message": "Invalid credentials"}, status_code=401)

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, admin_session: Optional[str] = Cookie(None)):
    if not verify_admin_session(admin_session):
        return RedirectResponse(url="/admin/login", status_code=302)
    
    cases = get_all_cases()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "cases": cases
    })

@app.post("/api/user/upload")
async def upload_missing_person(
    name: str = Form(...),
    age: int = Form(...),
    contact: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_filename = f"{name.replace(' ', '_')}_{image.filename}"
        image_path = f"uploads/images/{image_filename}"
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        case_id = create_case(name, age, contact, description, location, image_path)
        
        embedding = extract_face_embedding(image_path)
        
        if embedding is not None:
            update_case_embedding(case_id, embedding)
            status_message = "Case registered successfully. Face detected and ready for matching."
        else:
            status_message = "Case registered, but no face detected in the image. Please upload a clearer photo."
        
        return JSONResponse({
            "success": True,
            "case_id": case_id,
            "message": status_message
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error processing upload: {str(e)}"
        }, status_code=500)

@app.get("/api/user/status/{case_id}")
async def get_case_status(case_id: str):
    case = get_case(case_id)
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    matches = get_matches_for_case(case_id)
    
    response = {
        "case_id": case['case_id'],
        "name": case['name'],
        "age": case['age'],
        "status": case['status'],
        "location": case['location'],
        "match_place": case['match_place'],
        "matches": []
    }
    
    if matches:
        for match in matches:
            response['matches'].append({
                "frame_path": match['frame_path'],
                "timestamp": match['timestamp'],
                "similarity": round(match['similarity'], 3)
            })
    
    return response

@app.post("/api/admin/uploadVideo")
async def upload_video(
    video: UploadFile = File(...),
    location: str = Form(...),
    admin_session: Optional[str] = Cookie(None)
):
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        video_filename = video.filename
        video_path = f"uploads/videos/{video_filename}"
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        result = process_video(video_path, location)
        
        return JSONResponse({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error processing video: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/getResults/{case_id}")
async def get_results(case_id: str, admin_session: Optional[str] = Cookie(None)):
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    case = get_case(case_id)
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    matches = get_matches_for_case(case_id)
    
    return {
        "case": case,
        "matches": matches
    }

@app.post("/api/admin/confirmMatch/{case_id}")
async def confirm_match(case_id: str, admin_session: Optional[str] = Cookie(None)):
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        case = get_case(case_id)
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        update_case_status(case_id, "COMPLETED")
        
        return JSONResponse({
            "success": True,
            "message": "Match confirmed by police. Case marked as COMPLETED."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error confirming match: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/allCases")
async def get_all_cases_api(admin_session: Optional[str] = Cookie(None)):
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    cases = get_all_cases()
    return {"cases": cases}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
