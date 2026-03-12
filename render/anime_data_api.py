from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime
from pydantic import BaseModel, Field
import numpy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

anime_data_api = FastAPI(title= "Anime Score Predictor API")

anime_data_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    session = rt.InferenceSession('anime_predictor.onnx', providers=onnxruntime.get_available_providers())
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Error: Could not load ONNX model: {e}")
    session = None

class AnimeRequest(BaseModel):
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    rating: str = Field(..., example="PG-13")
    episodes: int = Field(default=12, ge=1, le=1000)

FEATURES = [
    'log_episodes',
    'mystery', 'suspense', 'sports', 'drama', 
    'slice_of_life', 'romance', 'adventure', 'supernatural', 'gourmet', 'action', 
    'fantasy', 'comedy', 'sci-fi', 'theme_iyashikei', 'theme_childcare', 
    'theme_gag_humor', 'theme_organized_crime', 'theme_adult_cast', 
    'theme_visual_arts', 'theme_love_status_quo', 'theme_love_polygon', 
    'theme_performing_arts', 'theme_combat_sports', 'theme_urban_fantasy', 
    'theme_cgdct', 'theme_team_sports', 'theme_otaku_culture', 'theme_showbiz', 
    'theme_historical', 'theme_time_travel', 'theme_racing', 'theme_delinquents', 
    'theme_detective', 'theme_medical', 'theme_military', 'theme_samurai', 
    'theme_school', 'rating_PG', 'rating_PG-13', 'rating_R', 'rating_R+', 'rating_Rx'
]

def preprocess(req: AnimeRequest):
    input_data = {col: 0.0 for col in FEATURES}

    input_data['log_episodes'] = numpy.log1p(req.episodes)

    for g in req.genres:
        if g in input_data: input_data[g] = 1.0
    for t in req.themes:
        if t in input_data: input_data[t] = 1.0
    rating_col = f"rating_{req.rating}"
    if rating_col in input_data:
        input_data[rating_col] = 1.0
    
    ordered_values = [input_data[col] for col in FEATURES]

    return numpy.array([list(ordered_values)], dtype=numpy.float32)


def get_prediction(data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: data})
    return result

@anime_data_api.get('/')
def root():
    return {'message': 'Welcome to the Anime Score Predictor API.'}

@app.get("/health")
def health():
    return {"status": "alive", "model_loaded": session is not None}

@anime_data_api.post("/predict")
async def predict_anime_score(req: AnimeRequest):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    try:
        input_tensor = preprocess(req)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: input_tensor})
        
        return {
            "predicted_score": float(prediction[0].item()),
            "status": "success"
        }
    except:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")