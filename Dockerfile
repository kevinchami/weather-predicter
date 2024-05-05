# Dockerfile

# Usar una imagen base de Python
FROM python:3.10

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos requeridos al contenedor
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY aysa_data.csv aysa_data.csv

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que correrá la API
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
