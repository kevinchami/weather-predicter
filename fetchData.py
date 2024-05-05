import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

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
train = daily_temp.loc[:'31/12/2021']
test = daily_temp.loc['01/01/2022':]
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
    
    # Predecir
    return reg.predict(new_data[predictors])[0]

# Predecir temperatura promedio para el 19/12/2024
future_date = "19/12/2024"
predicted_temp = predict_future(future_date, daily_temp, reg, predictors)
print(f"Predicción de la temperatura promedio para {future_date}: {predicted_temp:.2f} °C")
