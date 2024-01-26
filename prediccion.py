from flask import Flask, jsonify
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

env = 'http://bksedespotosi.appsbo.com/public/api/reportes'

def prediccion():
    base = '/dataset'
    response = requests.get(env+base)
    if response.status_code == 200:
        try:
            data_json = response.json()
            dataframe = pd.json_normalize(data_json)

            # Proceso de regresión lineal con el DataFrame
            X = dataframe[['semana']]
            y = dataframe['total']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')

            future_weeks = pd.DataFrame({'semana': range(13, 19)})
            future_predictions = model.predict(future_weeks)

            labels = pd.DataFrame(future_weeks)
            data = pd.DataFrame(future_predictions)
            return dataframe, labels, data
        except ValueError as e:
            print(f"Error al decodificar el JSON: {e}")
            return None, None
    else:
        return f'Error al hacer la solicitud: {response.status_code}'

@app.route("/")
def home():
    return "<h1> Hello this is Flask </h1>"

@app.route('/api/prediccion')
def regresion_lineal():
   dataset_resultado, labels, data = prediccion()
   if dataset_resultado is not None:
        formatted_labels = [f"SEMANA {i}" for i in labels['semana'].tolist()]

        # Convertir las predicciones a enteros sin redondear
        formatted_data = data.squeeze().astype(int).tolist()

        return jsonify({
            'labels': formatted_labels,
            'data': formatted_data
        })
   else:
       return 'Error al hacer la solicitud'

if __name__ =="__main__":
  app.run()
