import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones para forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from scipy import stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def check_forecasting_dependencies():
    """Verifica si las dependencias de forecasting están disponibles"""
    return STATSMODELS_AVAILABLE or SKLEARN_AVAILABLE

def prepare_forecast_data(df, column='valor'):
    """Prepara los datos para forecasting"""
    # Asegurar que tenemos una serie temporal continua
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[column])
    df_clean = df_clean.sort_values('fecha')
    
    # Crear índice de fechas
    df_clean.set_index('fecha', inplace=True)
    
    # Remover outliers extremos (opcional)
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Solo remover outliers muy extremos
    df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    return df_clean[column]

def simple_linear_forecast(series, periods=365):
    """Forecast simple usando regresión lineal"""
    # Crear variable de tiempo
    n = len(series)
    x = np.arange(n)
    y = series.values
    
    # División 80/20 para entrenamiento y prueba
    train_size = int(0.8 * n)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Entrenar modelo solo con datos de entrenamiento
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)
    
    # Predicciones en datos de prueba
    y_pred_test = slope * x_test + intercept
    
    # Calcular métricas de precisión en datos de prueba
    if len(y_test) > 0:
        metrics = calculate_forecast_metrics(y_test, y_pred_test)
    else:
        # Si no hay suficientes datos de prueba, usar todo el conjunto
        y_pred_train = slope * x + intercept
        metrics = calculate_forecast_metrics(y, y_pred_train)    
    # Generar fechas futuras
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Calcular predicciones futuras usando el modelo entrenado
    future_x = np.arange(n, n + periods)
    predictions = slope * future_x + intercept
    
    # Calcular intervalos de confianza usando residuos de entrenamiento
    y_pred_train_full = slope * x_train + intercept
    residuals = y_train - y_pred_train_full
    mse = np.mean(residuals**2)
    std_pred = np.sqrt(mse)
    
    confidence_interval = 1.96 * std_pred  # 95% de confianza
    
    forecast_df = pd.DataFrame({
        'fecha': future_dates,
        'prediccion': predictions,
        'limite_inferior': predictions - confidence_interval,
        'limite_superior': predictions + confidence_interval
    })
    
    return forecast_df, metrics

def seasonal_naive_forecast(series, periods=365):
    """Forecast usando método naive estacional (repite el patrón del año anterior)"""
    # Crear un DataFrame para trabajar más fácilmente
    df_series = pd.DataFrame({'fecha': series.index, 'valor': series.values})
    df_series['day_of_year'] = df_series['fecha'].dt.dayofyear
    
    # División 80/20 para entrenamiento y prueba
    n = len(df_series)
    train_size = int(0.8 * n)
    df_train = df_series[:train_size].copy()
    df_test = df_series[train_size:].copy()
    
    # Calcular promedios por día del año usando solo datos de entrenamiento
    seasonal_avg = df_train.groupby('day_of_year')['valor'].mean()
    
    # Calcular predicciones en datos de prueba
    if len(df_test) > 0:
        df_test['pred_seasonal'] = df_test['day_of_year'].map(seasonal_avg)
        # Manejar días que no están en el conjunto de entrenamiento
        df_test['pred_seasonal'] = df_test['pred_seasonal'].fillna(df_train['valor'].mean())
        
        # Calcular métricas de precisión en datos de prueba
        metrics = calculate_forecast_metrics(df_test['valor'].values, df_test['pred_seasonal'].values)
    else:
        # Si no hay suficientes datos de prueba, usar validación cruzada temporal
        df_series['pred_seasonal'] = df_series['day_of_year'].map(seasonal_avg)
        df_series['pred_seasonal'] = df_series['pred_seasonal'].fillna(df_series['valor'].mean())
        metrics = calculate_forecast_metrics(df_series['valor'].values, df_series['pred_seasonal'].values)
    
    # Generar fechas futuras
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Crear DataFrame para predicciones futuras
    future_df = pd.DataFrame({'fecha': future_dates})
    future_df['day_of_year'] = future_df['fecha'].dt.dayofyear
    
    # Manejar año bisiesto (día 366 -> usar día 365)
    future_df['day_of_year'] = future_df['day_of_year'].apply(lambda x: 365 if x == 366 else x)
    
    # Asignar predicciones
    future_df['prediccion'] = future_df['day_of_year'].map(seasonal_avg)
    
    # Calcular intervalos de confianza basados en la variabilidad histórica
    seasonal_std = df_series.groupby('day_of_year')['valor'].std().fillna(df_series['valor'].std())
    future_df['std'] = future_df['day_of_year'].map(seasonal_std)
    confidence_interval = 1.96 * future_df['std']
    
    future_df['limite_inferior'] = future_df['prediccion'] - confidence_interval
    future_df['limite_superior'] = future_df['prediccion'] + confidence_interval
    
    forecast_df = future_df[['fecha', 'prediccion', 'limite_inferior', 'limite_superior']]
    
    return forecast_df, metrics

def exponential_smoothing_forecast(series, periods=365):
    """Forecast usando Suavizado Exponencial (Holt-Winters)"""
    if not STATSMODELS_AVAILABLE:
        return None, "Statsmodels no está disponible", {}
    
    try:
        # División 80/20 para entrenamiento y prueba
        n = len(series)
        train_size = int(0.8 * n)
        series_train = series[:train_size]
        series_test = series[train_size:]
        
        # Entrenar modelo solo con datos de entrenamiento
        if len(series_train) >= 730:  # Al menos 2 años de datos de entrenamiento
            model = ExponentialSmoothing(
                series_train,
                trend='add',
                seasonal='add',
                seasonal_periods=365
            )
        else:
            # Sin estacionalidad si no hay suficientes datos
            model = ExponentialSmoothing(
                series_train,
                trend='add',
                seasonal=None
            )
        
        fitted_model = model.fit()
        
        # Evaluar en datos de prueba si están disponibles
        if len(series_test) > 0:
            # Hacer predicciones para el período de prueba
            test_forecast = fitted_model.forecast(steps=len(series_test))
            metrics = calculate_forecast_metrics(series_test.values, test_forecast.values)
        else:
            # Si no hay suficientes datos de prueba, usar ajuste en entrenamiento
            fitted_values = fitted_model.fittedvalues
            metrics = calculate_forecast_metrics(series_train.values, fitted_values.values)
        
        # Generar predicción final usando todo el conjunto de datos
        if len(series) >= 730:
            final_model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=365
            )
        else:
            final_model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None
            )
        
        final_fitted_model = final_model.fit()
        forecast = final_fitted_model.forecast(periods)        
        # Generar fechas futuras
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Calcular intervalos de confianza aproximados usando residuos de entrenamiento
        residuals = final_fitted_model.resid
        std_residuals = residuals.std()
        confidence_interval = 1.96 * std_residuals
        
        forecast_df = pd.DataFrame({
            'fecha': future_dates,
            'prediccion': forecast.values,
            'limite_inferior': forecast.values - confidence_interval,
            'limite_superior': forecast.values + confidence_interval
        })
        
        return forecast_df, f"AIC: {final_fitted_model.aic:.2f}, Precisión: {metrics['Precisión (%)']:.1f}%", metrics
        
    except Exception as e:
        return None, f"Error en Exponential Smoothing: {str(e)}", {}

def arima_forecast(series, periods=365):
    """Forecast usando modelo ARIMA"""
    if not STATSMODELS_AVAILABLE:
        return None, "Statsmodels no está disponible", {}
    
    try:
        # División 80/20 para entrenamiento y prueba
        n = len(series)
        train_size = int(0.8 * n)
        series_train = series[:train_size]
        series_test = series[train_size:]
        
        # Entrenar modelo ARIMA solo con datos de entrenamiento
        model = ARIMA(series_train, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Evaluar en datos de prueba si están disponibles
        if len(series_test) > 0:
            # Hacer predicciones para el período de prueba
            test_forecast = fitted_model.forecast(steps=len(series_test))
            metrics = calculate_forecast_metrics(series_test.values, test_forecast.values)
        else:
            # Si no hay suficientes datos de prueba, usar ajuste en entrenamiento
            fitted_values = fitted_model.fittedvalues
            # ARIMA puede tener algunos valores NaN al inicio, así que los manejamos
            valid_mask = ~np.isnan(fitted_values)
            if valid_mask.sum() > 0:
                metrics = calculate_forecast_metrics(
                    series_train[valid_mask].values, 
                    fitted_values[valid_mask].values
                )
            else:
                metrics = {'Precisión (%)': 0}
        
        # Entrenar modelo final con toda la serie para predicción futura
        final_model = ARIMA(series, order=(1, 1, 1))
        final_fitted_model = final_model.fit()        
        # Hacer predicción
        forecast = final_fitted_model.forecast(steps=periods)
        forecast_ci = final_fitted_model.get_prediction(start=len(series), end=len(series) + periods - 1).conf_int()
        
        # Generar fechas futuras
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'fecha': future_dates,
            'prediccion': forecast.values,
            'limite_inferior': forecast_ci.iloc[:, 0].values,
            'limite_superior': forecast_ci.iloc[:, 1].values
        })
        
        return forecast_df, f"AIC: {final_fitted_model.aic:.2f}, Precisión: {metrics['Precisión (%)']:.1f}%", metrics
        
    except Exception as e:
        return None, f"Error en ARIMA: {str(e)}", {}

def moving_average_forecast(series, periods=365, window=30):
    """Forecast usando promedio móvil"""
    # División 80/20 para entrenamiento y prueba
    n = len(series)
    train_size = int(0.8 * n)
    series_train = series[:train_size]
    series_test = series[train_size:]
    
    # Calcular promedio móvil en datos de entrenamiento
    ma_train = series_train.rolling(window=window).mean()
    
    # Calcular tendencia de los últimos valores de entrenamiento
    recent_ma = ma_train.tail(90)  # últimos 3 meses del entrenamiento
    if len(recent_ma) > 1:
        x = np.arange(len(recent_ma))
        slope, intercept, _, _, _ = stats.linregress(x, recent_ma.dropna().values)
        trend = slope
    else:
        trend = 0
    
    # Evaluar en datos de prueba si hay suficientes
    if len(series_test) > 0:
        # Extender el promedio móvil para datos de prueba
        last_ma = ma_train.dropna().iloc[-1] if len(ma_train.dropna()) > 0 else series_train.mean()
        
        # Generar predicciones para el período de prueba
        test_predictions = []
        for i in range(len(series_test)):
            pred = last_ma + trend * i
            test_predictions.append(pred)
        
        # Calcular métricas en datos de prueba
        metrics = calculate_forecast_metrics(series_test.values, np.array(test_predictions))
    else:
        # Si no hay suficientes datos de prueba, usar validación en ventana deslizante
        ma_filled = ma_train.bfill()
        valid_mask = ~ma_train.isna()
        if valid_mask.sum() > window:
            metrics = calculate_forecast_metrics(
                series_train[valid_mask].values, 
                ma_train[valid_mask].values
            )
        else:
            metrics = {'MAE': float('inf'), 'RMSE': float('inf'), 'R²': 0, 'Precisión (%)': 0}
    
    # Usar toda la serie para generar el modelo final
    ma_full = series.rolling(window=window).mean()
    last_ma = ma_full.dropna().iloc[-1] if len(ma_full.dropna()) > 0 else series.mean()
    
    # Generar fechas futuras
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Predicción con tendencia
    predictions = []
    for i in range(periods):
        pred = last_ma + trend * i
        predictions.append(pred)
    
    # Calcular intervalos de confianza
    recent_std = series.tail(window).std()
    confidence_interval = 1.96 * recent_std
    
    forecast_df = pd.DataFrame({
        'fecha': future_dates,
        'prediccion': predictions,
        'limite_inferior': np.array(predictions) - confidence_interval,
        'limite_superior': np.array(predictions) + confidence_interval
    })
    
    return forecast_df, metrics

def create_forecast_comparison_plot(historical_data, forecasts_dict, variable_name, unit):
    """Crea un gráfico comparativo de diferentes métodos de forecasting"""
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Datos Históricos',
        line=dict(color='blue', width=2)
    ))
    
    # Colores RGB para diferentes métodos
    colors = [
        'rgb(255,0,0)',    # rojo
        'rgb(0,128,0)',    # verde
        'rgb(255,165,0)',  # naranja
        'rgb(128,0,128)',  # púrpura
        'rgb(165,42,42)',  # marrón
        'rgb(255,20,147)', # rosa
        'rgb(0,191,255)',  # azul cielo
        'rgb(50,205,50)'   # verde lima
    ]
    
    for i, (method_name, forecast_df) in enumerate(forecasts_dict.items()):
        if forecast_df is not None:
            color = colors[i % len(colors)]
            
            # Extraer valores RGB del color para crear RGBA
            rgb_values = color.replace('rgb(', '').replace(')', '').split(',')
            r, g, b = [int(val.strip()) for val in rgb_values]
            rgba_color = f'rgba({r},{g},{b},0.2)'
            
            # Línea de predicción
            fig.add_trace(go.Scatter(
                x=forecast_df['fecha'],
                y=forecast_df['prediccion'],
                mode='lines',
                name=f'{method_name}',
                line=dict(color=color, width=2, dash='dash')
            ))
            
            # Intervalo de confianza
            if 'limite_superior' in forecast_df.columns and 'limite_inferior' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['fecha'].tolist() + forecast_df['fecha'].tolist()[::-1],
                    y=forecast_df['limite_superior'].tolist() + forecast_df['limite_inferior'].tolist()[::-1],
                    fill='toself',
                    fillcolor=rgba_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{method_name} - IC 95%',
                    showlegend=False
                ))
    
    fig.update_layout(
        title=f'Forecast de {variable_name} - Comparación de Métodos',
        xaxis_title='Fecha',
        yaxis_title=f'{variable_name} ({unit})',
        hovermode='x unified',
        height=600
    )
    
    return fig

def decompose_time_series(series, variable_name):
    """Descompone la serie temporal en tendencia, estacionalidad y residuos"""
    if len(series) < 730:  # Menos de 2 años
        st.warning("Se necesitan al menos 2 años de datos para una descomposición estacional completa")
        return None
    
    try:
        # Descomposición estacional
        decomposition = seasonal_decompose(series, model='additive', period=365)
        
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Serie Original', 'Tendencia', 'Estacionalidad', 'Residuos'),
            vertical_spacing=0.08
        )
        
        # Serie original
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Tendencia
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.trend, name='Tendencia', line=dict(color='red')),
            row=2, col=1
        )
        
        # Estacionalidad
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.seasonal, name='Estacionalidad', line=dict(color='green')),
            row=3, col=1
        )
        
        # Residuos
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.resid, name='Residuos', line=dict(color='orange')),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Descomposición de Serie Temporal - {variable_name}",
            showlegend=False
        )
        
        return fig, decomposition
        
    except Exception as e:
        st.error(f"Error en la descomposición: {str(e)}")
        return None, None

def calculate_forecast_metrics(actual, predicted):
    """Calcula métricas de evaluación del forecast"""
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
    
    # Convertir a arrays numpy para asegurar operaciones correctas
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Métricas básicas
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    # Evitar división por cero y valores muy pequeños
    non_zero_mask = np.abs(actual) > 1e-10
    if np.sum(non_zero_mask) > 0:
        mape_values = np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]) * 100
        mape = np.mean(mape_values)
        # Limitar MAPE a un valor máximo razonable
        mape = min(mape, 1000)
    else:
        mape = 100  # Si todos los valores son cero, asignar 100%
    
    # R² (Coeficiente de determinación)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
    
    # Precisión como porcentaje (basada en MAPE)
    if mape < 100:
        precision_pct = max(0, 100 - mape)
    else:
        precision_pct = 0
    
    # Error relativo medio
    actual_mean = np.mean(np.abs(actual))
    if actual_mean > 1e-10:
        mean_relative_error = np.mean(np.abs(actual - predicted)) / actual_mean * 100
    else:
        mean_relative_error = 100
    
    # Correlación de Pearson
    try:
        if len(actual) > 1 and np.std(actual) > 1e-10 and np.std(predicted) > 1e-10:
            correlation = np.corrcoef(actual, predicted)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
    except:
        correlation = 0
    
    # Accuracy dentro de cierto rango (±5%, ±10%, ±20%)
    actual_mean = np.mean(np.abs(actual))
    if actual_mean > 1e-10:
        tolerance_5 = 0.05 * actual_mean
        tolerance_10 = 0.10 * actual_mean
        tolerance_20 = 0.20 * actual_mean
        
        accuracy_5 = np.mean(np.abs(actual - predicted) <= tolerance_5) * 100
        accuracy_10 = np.mean(np.abs(actual - predicted) <= tolerance_10) * 100
        accuracy_20 = np.mean(np.abs(actual - predicted) <= tolerance_20) * 100
    else:
        accuracy_5 = accuracy_10 = accuracy_20 = 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Precisión (%)': precision_pct,
        'Error Relativo Medio (%)': mean_relative_error,
        'Correlación': correlation,
        'Accuracy ±5%': accuracy_5,
        'Accuracy ±10%': accuracy_10,
        'Accuracy ±20%': accuracy_20
    }

def run_forecast_module():
    """Función principal del módulo de forecasting"""
    st.title("🔮 Módulo de Predicción (Forecasting)")
    st.markdown("---")
    
    # Verificar dependencias
    if not check_forecasting_dependencies():
        st.warning("⚠️ Algunas funcionalidades avanzadas requieren statsmodels. Instala con: `pip install statsmodels`")
    
    # Cargar datos (reutilizar la función del dashboard principal)
    @st.cache_data
    def load_forecast_data():
        """Carga datos para forecasting"""
        import os
        datasets = {}
        datasets_path = "datasets_limpios"
        
        files = [
            "H34-Papallacta_Caudal-Diario.csv",
            "H34-Papallacta_Nivel_de_agua-Diario.csv",
            "M5025-La_Virgen_Papallacta_Dirección_de_viento-Diario.csv",
            "M5025-La_Virgen_Papallacta_Humedad_relativa-Diario.csv",
            "M5025-La_Virgen_Papallacta_Precipitación-Diario.csv",
            "M5025-La_Virgen_Papallacta_Presion_atmosférica-Diario.csv",
            "M5025-La_Virgen_Papallacta_Radiación_solar-Diario.csv",
            "M5025-La_Virgen_Papallacta_Temperatura_ambiente-Diario.csv",
            "M5025-La_Virgen_Papallacta_Velocidad_de_viento-Diario.csv"
        ]
        
        friendly_names = {
            "H34-Papallacta_Caudal-Diario.csv": "Caudal",
            "H34-Papallacta_Nivel_de_agua-Diario.csv": "Nivel de Agua",
            "M5025-La_Virgen_Papallacta_Dirección_de_viento-Diario.csv": "Dirección del Viento",
            "M5025-La_Virgen_Papallacta_Humedad_relativa-Diario.csv": "Humedad Relativa",
            "M5025-La_Virgen_Papallacta_Precipitación-Diario.csv": "Precipitación",
            "M5025-La_Virgen_Papallacta_Presion_atmosférica-Diario.csv": "Presión Atmosférica",
            "M5025-La_Virgen_Papallacta_Radiación_solar-Diario.csv": "Radiación Solar",
            "M5025-La_Virgen_Papallacta_Temperatura_ambiente-Diario.csv": "Temperatura Ambiente",
            "M5025-La_Virgen_Papallacta_Velocidad_de_viento-Diario.csv": "Velocidad del Viento"
        }
        
        units = {
            "Caudal": "m³/s",
            "Nivel de Agua": "m",
            "Dirección del Viento": "°",
            "Humedad Relativa": "%",
            "Precipitación": "mm",
            "Presión Atmosférica": "hPa",
            "Radiación Solar": "W/m²",
            "Temperatura Ambiente": "°C",
            "Velocidad del Viento": "m/s"
        }
        
        for file in files:
            try:
                df = pd.read_csv(os.path.join(datasets_path, file))
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.sort_values('fecha')
                
                name = friendly_names[file]
                datasets[name] = {
                    'data': df,
                    'unit': units[name]
                }
            except Exception as e:
                st.error(f"Error cargando {file}: {e}")
        
        return datasets
    
    # Cargar datos
    datasets = load_forecast_data()
    
    if not datasets:
        st.error("No se pudieron cargar los datos. Verifica que la carpeta 'datasets_limpios' existe y contiene los archivos CSV.")
        return
    
    # Sidebar para configuración
    st.sidebar.header("🎛️ Configuración de Predicción")
    
    # Selector de variable
    selected_variable = st.sidebar.selectbox(
        "Selecciona la variable a predecir:",
        list(datasets.keys())
    )
    
    # Periodo de predicción
    forecast_days = st.sidebar.slider(
        "Días a predecir:",
        min_value=30,
        max_value=730,
        value=365,
        step=30
    )
    
    # Métodos de forecasting    st.sidebar.subheader("Métodos de Predicción")
    use_linear = st.sidebar.checkbox("Regresión Lineal", value=True)
    use_seasonal = st.sidebar.checkbox("Naive Estacional", value=True)
    use_ma = st.sidebar.checkbox("Promedio Móvil", value=True)
    
    # Métodos avanzados con scikit-learn
    if SKLEARN_AVAILABLE:
        st.sidebar.subheader("Métodos Scikit-Learn")
        use_sklearn_linear = st.sidebar.checkbox("Regresión Lineal ML", value=False)
        use_sklearn_poly = st.sidebar.checkbox("Regresión Polinomial ML", value=False)
        if use_sklearn_poly:
            poly_degree = st.sidebar.slider("Grado del Polinomio", 2, 5, 3)
        use_sklearn_seasonal = st.sidebar.checkbox("Estacional ML", value=False)
    else:
        use_sklearn_linear = False
        use_sklearn_poly = False
        use_sklearn_seasonal = False
        st.sidebar.info("Instala scikit-learn para métodos de ML")
    
    # Métodos avanzados con statsmodels
    if STATSMODELS_AVAILABLE:
        st.sidebar.subheader("Métodos Statsmodels")
        use_exponential = st.sidebar.checkbox("Suavizado Exponencial", value=False)
        use_arima = st.sidebar.checkbox("ARIMA", value=False)
    else:
        use_exponential = False
        use_arima = False
        st.sidebar.info("Instala statsmodels para métodos avanzados")
    
    # Obtener datos de la variable seleccionada
    df = datasets[selected_variable]['data']
    unit = datasets[selected_variable]['unit']
    
    # Preparar serie temporal
    series = prepare_forecast_data(df)
    
    # Información de la serie
    st.subheader(f"📊 Información de la Serie - {selected_variable}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Observaciones", len(series))
    with col2:
        st.metric("Rango de Fechas", f"{(series.index[-1] - series.index[0]).days} días")
    with col3:
        st.metric("Valor Promedio", f"{series.mean():.2f} {unit}")
    with col4:
        st.metric("Desviación Estándar", f"{series.std():.2f} {unit}")
    
    # Descomposición de serie temporal
    if st.checkbox("📈 Mostrar Descomposición de Serie Temporal"):
        with st.spinner("Calculando descomposición..."):
            decomp_fig, decomposition = decompose_time_series(series, selected_variable)
            if decomp_fig:
                st.plotly_chart(decomp_fig, use_container_width=True)    
    # Ejecutar forecasting
    st.subheader("🔮 Predicciones")
    
    if st.button("Generar Predicciones", type="primary"):
        forecasts = {}
        forecast_info = {}
        forecast_metrics = {}
        
        with st.spinner("Generando predicciones..."):
            # Regresión lineal
            if use_linear:
                try:
                    forecast_df, metrics = simple_linear_forecast(series, forecast_days)
                    forecasts["Regresión Lineal"] = forecast_df
                    forecast_info["Regresión Lineal"] = f"R² = {metrics['R²']:.4f}"
                    forecast_metrics["Regresión Lineal"] = metrics
                except Exception as e:
                    st.error(f"Error en Regresión Lineal: {e}")
            
            # Naive estacional
            if use_seasonal:
                try:
                    forecast_df, metrics = seasonal_naive_forecast(series, forecast_days)
                    forecasts["Naive Estacional"] = forecast_df
                    forecast_info["Naive Estacional"] = f"Precisión = {metrics['Precisión (%)']:.1f}%"
                    forecast_metrics["Naive Estacional"] = metrics
                except Exception as e:
                    st.error(f"Error en Naive Estacional: {e}")              # Promedio móvil
            if use_ma:
                try:
                    forecast_df, metrics = moving_average_forecast(series, forecast_days)
                    forecasts["Promedio Móvil"] = forecast_df
                    forecast_info["Promedio Móvil"] = f"Ventana 30 días - Precisión = {metrics['Precisión (%)']:.1f}%"
                    forecast_metrics["Promedio Móvil"] = metrics
                except Exception as e:
                    st.error(f"Error en Promedio Móvil: {e}")
              # Métodos de scikit-learn
            if use_sklearn_linear:
                try:
                    forecast_df, info, metrics = sklearn_linear_forecast(df, forecast_days)
                    if forecast_df is not None:
                        forecasts["ML Linear"] = forecast_df
                        forecast_info["ML Linear"] = info
                        forecast_metrics["ML Linear"] = metrics
                except Exception as e:
                    st.error(f"Error en ML Linear: {e}")
            
            if use_sklearn_poly:
                try:
                    forecast_df, info, metrics = sklearn_polynomial_forecast(df, forecast_days, poly_degree)
                    if forecast_df is not None:
                        forecasts[f"ML Polinomial (grado {poly_degree})"] = forecast_df
                        forecast_info[f"ML Polinomial (grado {poly_degree})"] = info
                        forecast_metrics[f"ML Polinomial (grado {poly_degree})"] = metrics
                except Exception as e:
                    st.error(f"Error en ML Polinomial: {e}")
            
            if use_sklearn_seasonal:
                try:
                    forecast_df, info, metrics = sklearn_seasonal_forecast(df, forecast_days)
                    if forecast_df is not None:
                        forecasts["ML Estacional"] = forecast_df
                        forecast_info["ML Estacional"] = info
                        forecast_metrics["ML Estacional"] = metrics
                except Exception as e:
                    st.error(f"Error en ML Estacional: {e}")
            
            # Suavizado exponencial
            if use_exponential:
                try:
                    forecast_df, info, metrics = exponential_smoothing_forecast(series, forecast_days)
                    if forecast_df is not None:
                        forecasts["Suavizado Exponencial"] = forecast_df
                        forecast_info["Suavizado Exponencial"] = info
                        forecast_metrics["Suavizado Exponencial"] = metrics
                    else:
                        st.error(info)
                except Exception as e:
                    st.error(f"Error en Suavizado Exponencial: {e}")
            
            # ARIMA
            if use_arima:
                try:
                    forecast_df, info, metrics = arima_forecast(series, forecast_days)
                    if forecast_df is not None:
                        forecasts["ARIMA"] = forecast_df
                        forecast_info["ARIMA"] = info
                        forecast_metrics["ARIMA"] = metrics
                    else:
                        st.error(info)
                except Exception as e:
                    st.error(f"Error en ARIMA: {e}")
          # Mostrar resultados
        if forecasts:
            # Gráfico comparativo
            fig = create_forecast_comparison_plot(series, forecasts, selected_variable, unit)
            st.plotly_chart(fig, use_container_width=True)
            
            # Información de los métodos
            st.subheader("📋 Información de los Métodos")
            for method, info in forecast_info.items():
                st.write(f"**{method}:** {info}")
              # Métricas de precisión detalladas
            if forecast_metrics:
                st.subheader("📊 Métricas de Precisión (evaluadas en datos de prueba - 20% del conjunto)")
                st.info("💡 **Nota importante**: Las métricas mostradas se calculan usando validación 80/20 (80% entrenamiento, 20% prueba) para evaluar la verdadera precisión del modelo.")
                
                  # Crear DataFrame con todas las métricas
                metrics_data = []
                for method, metrics in forecast_metrics.items():
                    # Formatear métricas con manejo de valores especiales
                    mae_val = metrics.get('MAE', 0)
                    rmse_val = metrics.get('RMSE', 0)
                    mape_val = metrics.get('MAPE', 0)
                    r2_val = metrics.get('R²', 0)
                    precision_val = metrics.get('Precisión (%)', 0)
                    correlation_val = metrics.get('Correlación', 0)
                    
                    metrics_data.append({
                        'Método': method,
                        'MAE': f"{mae_val:.4f}" if mae_val < 1e6 else "N/A",
                        'RMSE': f"{rmse_val:.4f}" if rmse_val < 1e6 else "N/A",
                        'MAPE (%)': f"{mape_val:.2f}%" if mape_val < 1000 else ">1000%",
                        'R²': f"{r2_val:.4f}" if abs(r2_val) < 100 else "N/A",
                        'Precisión (%)': f"{precision_val:.1f}%",
                        'Correlación': f"{correlation_val:.4f}" if abs(correlation_val) <= 1 else "N/A"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Métricas adicionales (accuracy) en expander
                with st.expander("🎯 Métricas de Precisión Adicionales"):
                    accuracy_data = []
                    for method, metrics in forecast_metrics.items():
                        accuracy_data.append({
                            'Método': method,
                            'Accuracy ±5%': f"{metrics.get('Accuracy ±5%', 0):.1f}%",
                            'Accuracy ±10%': f"{metrics.get('Accuracy ±10%', 0):.1f}%",
                            'Accuracy ±20%': f"{metrics.get('Accuracy ±20%', 0):.1f}%",
                            'Error Relativo Medio (%)': f"{metrics.get('Error Relativo Medio (%)', 0):.2f}%"
                        })
                    
                    accuracy_df = pd.DataFrame(accuracy_data)
                    st.dataframe(accuracy_df, use_container_width=True)
                    
                    st.caption("""
                    **Explicación de métricas:**
                    - **MAE**: Error Absoluto Medio (menor es mejor)
                    - **RMSE**: Raíz del Error Cuadrático Medio (menor es mejor)
                    - **MAPE**: Error Porcentual Absoluto Medio (menor es mejor)
                    - **R²**: Coeficiente de determinación (más cercano a 1 es mejor)
                    - **Precisión (%)**: Porcentaje de precisión basado en MAPE
                    - **Correlación**: Correlación de Pearson entre valores reales y predichos
                    - **Accuracy ±X%**: Porcentaje de predicciones dentro del rango de tolerancia
                    """)
            
            # Estadísticas de las predicciones
            st.subheader("� Estadísticas de las Predicciones Futuras")
            
            stats_data = []
            for method, forecast_df in forecasts.items():
                stats_data.append({
                    'Método': method,
                    'Promedio': f"{forecast_df['prediccion'].mean():.3f}",
                    'Mínimo': f"{forecast_df['prediccion'].min():.3f}",
                    'Máximo': f"{forecast_df['prediccion'].max():.3f}",
                    'Desv. Estándar': f"{forecast_df['prediccion'].std():.3f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Opción de descarga
            st.subheader("💾 Descargar Predicciones")
            
            # Combinar todas las predicciones en un solo DataFrame
            combined_forecasts = pd.DataFrame({'fecha': forecasts[list(forecasts.keys())[0]]['fecha']})
            
            for method, forecast_df in forecasts.items():
                combined_forecasts[f'{method}_prediccion'] = forecast_df['prediccion']
                combined_forecasts[f'{method}_limite_inferior'] = forecast_df['limite_inferior']
                combined_forecasts[f'{method}_limite_superior'] = forecast_df['limite_superior']
            
            csv = combined_forecasts.to_csv(index=False)
            st.download_button(
                label="Descargar predicciones como CSV",
                data=csv,
                file_name=f"predicciones_{selected_variable}_{forecast_days}dias.csv",
                mime="text/csv"
            )
        else:
            st.error("No se pudieron generar predicciones. Verifica los datos y la configuración.")

if __name__ == "__main__":
    run_forecast_module()

def prepare_sklearn_features(df, target_col='valor'):
    """Prepara características para scikit-learn"""
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Características temporales
    df_clean['year'] = df_clean['fecha'].dt.year
    df_clean['month'] = df_clean['fecha'].dt.month
    df_clean['day'] = df_clean['fecha'].dt.day
    df_clean['day_of_year'] = df_clean['fecha'].dt.dayofyear
    df_clean['day_of_week'] = df_clean['fecha'].dt.dayofweek
    
    # Variables numéricas (días desde el inicio)
    start_date = df_clean['fecha'].min()
    df_clean['days_since_start'] = (df_clean['fecha'] - start_date).dt.days
    
    # Características cíclicas (para capturar estacionalidad)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
    df_clean['day_sin'] = np.sin(2 * np.pi * df_clean['day_of_year'] / 365.25)
    df_clean['day_cos'] = np.cos(2 * np.pi * df_clean['day_of_year'] / 365.25)
    
    return df_clean

def sklearn_linear_forecast(df, periods=365, target_col='valor'):
    """Forecasting con regresión lineal usando scikit-learn"""
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn no disponible", {}
    
    df_prepared = prepare_sklearn_features(df, target_col)
    
    # Preparar características
    feature_cols = ['days_since_start', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    X = df_prepared[feature_cols].values
    y = df_prepared[target_col].values
    
    # División 80/20 para entrenamiento y prueba
    n = len(X)
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Entrenar modelo solo con datos de entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones en datos de prueba
    if len(X_test) > 0:
        y_pred_test = model.predict(X_test)
        metrics = calculate_forecast_metrics(y_test, y_pred_test)
    else:
        # Si no hay suficientes datos de prueba, usar todo el conjunto
        y_pred_all = model.predict(X)
        metrics = calculate_forecast_metrics(y, y_pred_all)    
    # Generar características para el futuro
    last_date = df_prepared['fecha'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Crear DataFrame para el futuro
    future_df = pd.DataFrame({'fecha': future_dates})
    future_df['month'] = future_df['fecha'].dt.month
    future_df['day_of_year'] = future_df['fecha'].dt.dayofyear
    
    start_date = df_prepared['fecha'].min()
    future_df['days_since_start'] = (future_df['fecha'] - start_date).dt.days
    
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365.25)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365.25)
    
    X_future = future_df[feature_cols].values
    
    # Hacer predicciones
    predictions = model.predict(X_future)
    
    # Calcular intervalos de confianza usando residuos de entrenamiento
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error
    
    forecast_df = pd.DataFrame({
        'fecha': future_dates,
        'prediccion': predictions,
        'limite_inferior': predictions - confidence_interval,
        'limite_superior': predictions + confidence_interval
    })
    
    return forecast_df, f"Regresión Lineal ML (R² = {metrics['R²']:.3f}, Precisión = {metrics['Precisión (%)']:.1f}%)", metrics

def sklearn_polynomial_forecast(df, periods=365, degree=3, target_col='valor'):
    """Forecasting con regresión polinomial usando scikit-learn"""
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn no disponible", {}
    
    df_prepared = prepare_sklearn_features(df, target_col)
    
    # Preparar características base
    feature_cols = ['days_since_start', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    X_base = df_prepared[feature_cols].values
    
    # Crear características polinomiales
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_base)
    
    y = df_prepared[target_col].values
    
    # División 80/20 para entrenamiento y prueba
    n = len(X_poly)
    train_size = int(0.8 * n)
    X_train, X_test = X_poly[:train_size], X_poly[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Entrenar modelo solo con datos de entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluación en datos de prueba
    if len(X_test) > 0:
        y_pred_test = model.predict(X_test)
        metrics = calculate_forecast_metrics(y_test, y_pred_test)
    else:
        # Si no hay suficientes datos de prueba, usar todo el conjunto
        y_pred_all = model.predict(X_poly)
        metrics = calculate_forecast_metrics(y, y_pred_all)    
    # Preparar datos futuros
    last_date = df_prepared['fecha'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    future_df = pd.DataFrame({'fecha': future_dates})
    future_df['month'] = future_df['fecha'].dt.month
    future_df['day_of_year'] = future_df['fecha'].dt.dayofyear
    
    start_date = df_prepared['fecha'].min()
    future_df['days_since_start'] = (future_df['fecha'] - start_date).dt.days
    
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365.25)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365.25)
    
    X_future_base = future_df[feature_cols].values
    X_future_poly = poly_features.transform(X_future_base)
    
    # Predicciones
    predictions = model.predict(X_future_poly)
    
    # Intervalos de confianza usando residuos de entrenamiento
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error
    
    forecast_df = pd.DataFrame({
        'fecha': future_dates,
        'prediccion': predictions,
        'limite_inferior': predictions - confidence_interval,
        'limite_superior': predictions + confidence_interval
    })
    
    return forecast_df, f"Regresión Polinomial grado {degree} (R² = {metrics['R²']:.3f}, Precisión = {metrics['Precisión (%)']:.1f}%)", metrics

def sklearn_seasonal_forecast(df, periods=365, target_col='valor'):
    """Forecasting estacional usando promedios por día del año"""
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean['day_of_year'] = df_clean['fecha'].dt.dayofyear
    
    # División 80/20 para entrenamiento y prueba
    n = len(df_clean)
    train_size = int(0.8 * n)
    df_train = df_clean[:train_size].copy()
    df_test = df_clean[train_size:].copy()
    
    # Calcular promedios por día del año usando solo datos de entrenamiento
    seasonal_avg = df_train.groupby('day_of_year')[target_col].mean()
    seasonal_std = df_train.groupby('day_of_year')[target_col].std().fillna(0)
    
    # Calcular métricas usando datos de prueba
    if len(df_test) > 0:
        df_test['pred_seasonal'] = df_test['day_of_year'].map(seasonal_avg)
        # Manejar días que no están en el conjunto de entrenamiento
        df_test['pred_seasonal'] = df_test['pred_seasonal'].fillna(df_train[target_col].mean())
        metrics = calculate_forecast_metrics(df_test[target_col].values, df_test['pred_seasonal'].values)
    else:
        # Si no hay suficientes datos de prueba, usar validación cruzada
        df_clean['pred_seasonal'] = df_clean['day_of_year'].map(seasonal_avg)
        df_clean['pred_seasonal'] = df_clean['pred_seasonal'].fillna(df_clean[target_col].mean())
        metrics = calculate_forecast_metrics(df_clean[target_col].values, df_clean['pred_seasonal'].values)    
    # Generar fechas futuras
    last_date = df_clean['fecha'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    future_df = pd.DataFrame({'fecha': future_dates})
    future_df['day_of_year'] = future_df['fecha'].dt.dayofyear
    
    # Manejar año bisiesto (día 366 -> usar día 365)
    future_df['day_of_year'] = future_df['day_of_year'].apply(lambda x: 365 if x == 366 else x)
    
    # Asignar predicciones usando los promedios calculados en entrenamiento
    future_df['prediccion'] = future_df['day_of_year'].map(seasonal_avg)
    future_df['std'] = future_df['day_of_year'].map(seasonal_std)
    
    # Intervalos de confianza basados en la variabilidad histórica
    future_df['limite_inferior'] = future_df['prediccion'] - 1.96 * future_df['std']
    future_df['limite_superior'] = future_df['prediccion'] + 1.96 * future_df['std']
    
    forecast_df = future_df[['fecha', 'prediccion', 'limite_inferior', 'limite_superior']]
    
    return forecast_df, f"Forecast Estacional (R² = {metrics['R²']:.3f}, Precisión = {metrics['Precisión (%)']:.1f}%)", metrics
