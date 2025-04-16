# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.game import router as game_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay bằng domain của frontend (ví dụ: "https://ataxx-hex.netlify.app")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router)

@app.get("/")
async def root():
    return {"message": "Ataxx Backend API"}