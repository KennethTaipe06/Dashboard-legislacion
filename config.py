# Configuración del Sistema de Análisis Meteorológico

# Configuración de datos
DATA_PATH = "datasets_limpios"
ORIGINAL_DATA_PATH = "datasets"

# Configuración de forecasting
DEFAULT_FORECAST_DAYS = 365
MAX_FORECAST_DAYS = 730
MIN_FORECAST_DAYS = 30

# Configuración de gráficos
DEFAULT_PLOT_HEIGHT = 500
DEFAULT_PLOT_WIDTH = None  # Usa el ancho completo

# Colores para diferentes métodos de forecasting
FORECAST_COLORS = {
    'Regresión Lineal': 'red',
    'Naive Estacional': 'green', 
    'Promedio Móvil': 'orange',
    'Suavizado Exponencial': 'purple',
    'ARIMA': 'brown'
}

# Configuración de limpieza de datos
OUTLIER_THRESHOLD = 3  # Número de IQRs para considerar outlier
MIN_DATA_POINTS = 30   # Mínimo de puntos de datos para análisis

# Configuración de series temporales
SEASONAL_PERIOD = 365  # Días por año para análisis estacional
MOVING_AVERAGE_WINDOW = 30  # Ventana por defecto para promedio móvil

# Mensajes de información
INFO_MESSAGES = {
    'insufficient_data': "Se necesitan al menos {min_points} puntos de datos para el análisis",
    'missing_dependencies': "Instala statsmodels para métodos avanzados de forecasting: pip install statsmodels",
    'forecast_complete': "Predicciones generadas exitosamente",
    'data_cleaned': "Datos limpiados y validados automáticamente"
}

# Configuración de exportación
EXPORT_DATE_FORMAT = "%Y-%m-%d"
CSV_SEPARATOR = ","
