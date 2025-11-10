from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from .database import SessionLocal, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
tokens = {}  # in-memory token store

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    username = tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    return user
