from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Inicializar FastAPI
app = FastAPI()

# Leer los datos
weather = pd.read_csv('aysa_data.csv', index_col="Fecha", dayfirst=True)
weather.index = pd.to_datetime(weather.index, dayfirst=True)
core_weather = weather[["Temperatura", "Humedad"]].copy()

# Limpiar los datos
core_weather["Temperatura"].fillna(method='ffill', inplace=True)
core_weather["Humedad"].fillna(method='ffill', inplace=True)
core_weather["Temperatura"] = core_weather["Temperatura"].str.replace(',', '.').astype(float)
core_weather["Humedad"] = core_weather["Humedad"].str.replace(',', '.').astype(float)

# Resample para obtener una temperatura y humedad por día
daily_temp = core_weather.between_time("14:00", "14:00").resample('D').first()
daily_temp["target"] = daily_temp["Temperatura"].shift(-1)
daily_temp = daily_temp.iloc[:-1, :].copy()

# Calcular características adicionales
daily_temp["month_max"] = daily_temp["Temperatura"].rolling(30).max()
daily_temp["month_min"] = daily_temp["Temperatura"].rolling(30).min()
daily_temp["month_day_max"] = daily_temp["month_max"] / daily_temp["Temperatura"]

# Añadir medias expandibles
def expanding_monthly_mean(group):
    return group.expanding().mean()

def expanding_day_of_year_mean(group):
    return group.expanding().mean()

monthly_avg = daily_temp["Temperatura"].groupby(daily_temp.index.to_period('M')).apply(expanding_monthly_mean)
monthly_avg.index = monthly_avg.index.droplevel(0)
daily_temp.loc[:, "monthly_avg"] = monthly_avg

day_of_year_avg = daily_temp["Temperatura"].groupby(daily_temp.index.day_of_year).apply(expanding_day_of_year_mean)
day_of_year_avg.index = day_of_year_avg.index.droplevel(0)
daily_temp.loc[:, "day_of_year_avg"] = day_of_year_avg

# Definir los predictores
predictors = ["Temperatura", "Humedad", "month_max", "monthly_avg", "day_of_year_avg"]

# Entrenar el modelo
reg = Ridge(alpha=.1)
train = daily_temp.loc[:'2021-12-31']
test = daily_temp.loc['2022-01-01':]
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)
train.dropna(inplace=True)
reg.fit(train[predictors], train["target"])

# Función para predecir en una fecha futura
def predict_future(date, daily_temp, reg, predictors):
    date = pd.to_datetime(date, dayfirst=True)
    new_data = pd.DataFrame(index=[date])
    
    # Establecer características
    new_data["Temperatura"] = daily_temp["Temperatura"].iloc[-1]
    new_data["Humedad"] = daily_temp["Humedad"].iloc[-1]
    new_data["month_max"] = daily_temp["Temperatura"].rolling(30).max().iloc[-1]
    new_data["month_day_max"] = new_data["month_max"] / new_data["Temperatura"]
    
    # Añadir características adicionales
    new_data["monthly_avg"] = daily_temp["Temperatura"].groupby(daily_temp.index.to_period('M')).mean().iloc[-1]
    day_of_year_avg = daily_temp["Temperatura"].groupby(daily_temp.index.day_of_year).mean()
    new_data["day_of_year_avg"] = day_of_year_avg.loc[date.day_of_year] if date.day_of_year in day_of_year_avg.index else day_of_year_avg.mean()
    
    # Rellenar valores faltantes con valores anteriores
    new_data.fillna(method='ffill', inplace=True)
    
    # Predecir
    return reg.predict(new_data[predictors])[0]

# Modelo Pydantic para el request
class DateRequest(BaseModel):
    date: str

@app.post("/predict_temperature/")
def predict_temperature(request: DateRequest):
    try:
        predicted_temp = predict_future(request.date, daily_temp, reg, predictors)
        return {"date": request.date, "predicted_temperature": round(predicted_temp, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test_model/")
def test_model():
    try:
        # Generar predicciones para los datos de test
        predictions = reg.predict(test[predictors])
        error = mean_absolute_error(test["target"], predictions)
        
        # Combinar resultados reales y predicciones
        combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
        combined.columns = ["actual", "predicted"]
        
        # Convertir resultados a JSON
        results = combined.reset_index().to_dict(orient="records")
        
        return {
            "mean_absolute_error": round(error, 2),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
