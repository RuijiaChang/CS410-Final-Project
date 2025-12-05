from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import api

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommender System API"}
