from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from fastapi.templating import Jinja2Templates

# Environment settings
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template directory
templates = Jinja2Templates(directory="templates")

# Configuration class for the client
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

# Pydantic model for JSON input
class ImageRequest(BaseModel):
    image: str

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home route that renders the HTML file.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
async def train_route():
    """
    Route to train the model by running a script.
    """
    os.system("python main.py")
    return {"message": "Training done successfully!"}

@app.post("/predict")
async def predict_route(image_request: ImageRequest):
    """
    Route to make predictions based on the provided image.
    """
    try:
        decodeImage(image_request.image, clApp.filename)
        result = clApp.classifier.predict()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
