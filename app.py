# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from wsd_core import disambiguate

app = FastAPI(title="Tamil WSD API", version="1.0.0")

class WsdIn(BaseModel):
    text_ta: str

class WsdOut(BaseModel):
    target_word: str | None
    best_sense: str | None
    gloss_ta: str | None
    scores: dict
    bag_words: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/wsd", response_model=WsdOut)
def wsd(req: WsdIn):
    result = disambiguate(req.text_ta.strip())
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
