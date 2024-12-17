from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from fastapi.templating import Jinja2Templates

# Configuraciones de entorno
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Inicializar FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir los orígenes en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio de plantillas
templates = Jinja2Templates(directory="templates")

# Clase de configuración para el cliente
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

# Modelo Pydantic para la entrada JSON
class ImageRequest(BaseModel):
    image: str

# Rutas FastAPI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Ruta de inicio que renderiza el archivo HTML.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
async def train_route():
    """
    Ruta para entrenar el modelo ejecutando un script.
    """
    os.system("python main.py")
    return {"message": "Training done successfully!"}

@app.post("/predict")
async def predict_route(image_request: ImageRequest):
    """
    Ruta para realizar predicciones basadas en la imagen enviada.
    """
    try:
        decodeImage(image_request.image, clApp.filename)
        result = clApp.classifier.predict()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

# Punto de entrada principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)