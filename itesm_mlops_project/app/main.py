import logging
import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import ModelPredictor
from models.models import Titanic

logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.DEBUG) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('main_api.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo


app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Titanic classifier is all ready to go!")
    return 'Titanic classifier is all ready to go!'

@app.post('/predict')
def predict(titanic_features: Titanic) -> JSONResponse:
    predictor = ModelPredictor("ml_models/logistic_regression_output.pkl")
    X = [titanic_features.pclass_nan,
        titanic_features.age_nan,
        titanic_features.sibsp_nan,
        titanic_features.parch_nan,
        titanic_features.fare_nan,
        titanic_features.sex_male,
        titanic_features.cabin_Missing,
        titanic_features.cabin_rare,
        titanic_features.embarked_Q,
        titanic_features.embarked_S,
        titanic_features.title_Mr,
        titanic_features.title_Mrs,
        titanic_features.title_rar]
    print(f"Input values: {[X]}")
    logger.info(f"Input values: {[X]}")
    prediction = predictor.predict([X])
    logger.info(f"Resultado predicción: {prediction}")
    return JSONResponse(f"Resultado predicción: {prediction}")
