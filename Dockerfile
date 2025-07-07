# Usar Python 3.11 como imagen base
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios si no existen
RUN mkdir -p datasets_limpios datasets

# Exponer el puerto donde correrá Streamlit
EXPOSE 8503

# Configurar variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8503
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "main.py", "--server.port=8503", "--server.address=0.0.0.0"]
