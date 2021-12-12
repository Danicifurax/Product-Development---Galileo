import joblib
import config as cfg
import pandas as pd
import numpy as np

#Cargamos modelo y pipeline
titanic_model = joblib.load('titanic_pipeline.pkl')

#Funcion para hacer predicciones.
def predict(X):
    predicts = titanic_model.predict(X)
    salida = np.exp(predicts)
    print(salida)
    return salida[0]