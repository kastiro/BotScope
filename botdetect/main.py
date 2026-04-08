"""BotScope - Bot Detection Frontend Server."""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI(title="BotScope")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


@app.get("/")
def index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))


@app.get("/user-analysis")
def user_analysis():
    return FileResponse(os.path.join(TEMPLATES_DIR, "bot_detector.html"))


@app.get("/hashtag-analysis")
def hashtag_analysis():
    return FileResponse(os.path.join(TEMPLATES_DIR, "analyzer.html"))


@app.get("/simulation")
def simulation():
    return FileResponse(os.path.join(TEMPLATES_DIR, "demo.html"))


if __name__ == "__main__":
    print("BotScope running at http://localhost:4000")
    uvicorn.run(app, host="0.0.0.0", port=4000)
