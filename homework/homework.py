#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

"""Entrena y evalúa un modelo de predicción de precios de vehículos usados."""

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures

INPUT_DIR = "files/input"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TARGET = "Present_Price"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables requeridas por el modelo y elimina las no usadas."""
    prepared = data.copy()
    prepared["Age"] = 2021 - prepared["Year"]
    return prepared.drop(columns=["Year", "Car_Name"])


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la variable objetivo de los predictores."""
    return data.drop(columns=[TARGET]), data[TARGET]


def build_model() -> GridSearchCV:
    """Construye la búsqueda con preprocesamiento, selección y regresión."""
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = ["Selling_Price", "Driven_kms", "Owner", "Age"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", MinMaxScaler(), numerical_features),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("polynomial_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("select_k_best", SelectKBest(score_func=f_regression)),
        ("linear_regression", LinearRegression()),
    ])
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"select_k_best__k": range(1, 21)},
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )


def metric_record(dataset: str, y_true: pd.Series, y_pred: object) -> dict[str, object]:
    """Devuelve las métricas solicitadas para un conjunto de datos."""
    return {
        "type": "metrics",
        "dataset": dataset,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": mean_absolute_error(y_true, y_pred),
    }


def main() -> None:
    """Ejecuta entrenamiento, persistencia y evaluación."""
    train = prepare_data(pd.read_csv(f"{INPUT_DIR}/train_data.csv.zip"))
    test = prepare_data(pd.read_csv(f"{INPUT_DIR}/test_data.csv.zip"))
    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)
    model = build_model()
    model.fit(x_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with gzip.open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    metrics = [
        metric_record("train", y_train, model.predict(x_train)),
        metric_record("test", y_test, model.predict(x_test)),
    ]
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        for record in metrics:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
