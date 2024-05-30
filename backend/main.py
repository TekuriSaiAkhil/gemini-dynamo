from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"]
)

genai_processor = GeminiProcessor(model_name="gemini-pro", project="gemini-dynamo-424721")

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):

    processor = YoutubeProcessor(genai_processor = genai_processor)
    result = processor.retrive_youtube_documents(str(request.youtube_link), verbose=True)
    key_concepts = processor.get_key_concepts(result, sample_size=8, verbose=True)

    return {
        "key_concepts" : key_concepts
    }

@app.get("/root")
def health():
    return {"status": "ok"}