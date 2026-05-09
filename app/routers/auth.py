from fastapi import APIRouter, Request, Depends, Form, responses
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app.models import User, Base
from app.auth import get_password_hash, verify_password, create_access_token

# Create tables
Base.metadata.create_all(bind=engine)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/signup")
def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post("/signup")
def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    
    # Check if user exists
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Email already registered"})
    
    # Create user
    new_user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password)
    )
    db.add(new_user)
    db.commit()
    
    return responses.RedirectResponse(url="/login", status_code=303)

@router.get("/login")
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    
    # Find user by username or email
    user = db.query(User).filter((User.username == username) | (User.email == username)).first()
    
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    
    # Create Token
    access_token = create_access_token(data={"sub": user.username})
    
    # Set cookie and redirect
    response = responses.RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@router.get("/logout")
def logout():
    response = responses.RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response