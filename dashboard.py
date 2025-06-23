import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones opcionales para funcionalidades avanzadas
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Meteorológico e Hidrológico - Papallacta",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌦️ Dashboard Meteorológico e Hidrológico - Papallacta")
st.markdown("---")

# Definir nombres de meses globalmente
month_names = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

# Función para cargar datos
@st.cache_data
def load_data():
    """Carga todos los datasets desde la carpeta datasets_limpios"""
    datasets = {}
    datasets_path = "datasets_limpios"
    
    # Lista de archivos en la carpeta
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
    
    # Mapeo de nombres amigables
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
    
    # Unidades para cada variable
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
datasets = load_data()

# Sidebar para selección
st.sidebar.header("🎛️ Panel de Control")

# Selector de dataset
selected_dataset = st.sidebar.selectbox(
    "Selecciona una variable:",
    list(datasets.keys()),
    index=0
)

# Información del dataset seleccionado
if selected_dataset in datasets:
    df = datasets[selected_dataset]['data']
    unit = datasets[selected_dataset]['unit']
    
    st.sidebar.markdown(f"**Dataset:** {selected_dataset}")
    st.sidebar.markdown(f"**Unidad:** {unit}")
    st.sidebar.markdown(f"**Registros:** {len(df)}")
    st.sidebar.markdown(f"**Período:** {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}")

# Filtro de fechas
st.sidebar.subheader("📅 Filtro de Fechas")
if selected_dataset in datasets:
    df = datasets[selected_dataset]['data']
    
    min_date = df['fecha'].min().date()
    max_date = df['fecha'].max().date()
    
    date_range = st.sidebar.date_input(
        "Selecciona el rango de fechas:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtrar datos por fecha
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['fecha'].dt.date >= start_date) & (df['fecha'].dt.date <= end_date)]
    else:
        df_filtered = df

# Tipo de visualización
st.sidebar.subheader("📊 Tipo de Gráfico")
chart_type = st.sidebar.selectbox(
    "Selecciona el tipo de visualización:",
    ["Serie Temporal", "Histograma", "Box Plot", "Estadísticas", "Análisis de Tendencia", "Comparación Anual"]
)

# Contenido principal
if selected_dataset in datasets:
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"Valor Promedio ({unit})",
            value=f"{df_filtered['valor'].mean():.2f}",
            delta=f"{df_filtered['valor'].mean() - df['valor'].mean():.2f}"
        )
    
    with col2:
        st.metric(
            label=f"Máximo ({unit})",
            value=f"{df_filtered['valor'].max():.2f}"
        )
    
    with col3:
        st.metric(
            label=f"Mínimo ({unit})",
            value=f"{df_filtered['valor'].min():.2f}"
        )
    
    with col4:
        st.metric(
            label="Completitud (%)",
            value=f"{df_filtered['completo_mediciones'].mean():.1f}%"
        )
    
    st.markdown("---")
    
    # Visualizaciones según el tipo seleccionado
    if chart_type == "Serie Temporal":
        st.subheader(f"📈 Serie Temporal - {selected_dataset}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['fecha'],
            y=df_filtered['valor'],
            mode='lines',
            name=selected_dataset,
            line=dict(color='#1f77b4', width=2)
        ))
        
        if 'max_abs' in df_filtered.columns and 'min_abs' in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered['fecha'],
                y=df_filtered['max_abs'],
                mode='lines',
                name='Máximo Absoluto',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df_filtered['fecha'],
                y=df_filtered['min_abs'],
                mode='lines',
                name='Mínimo Absoluto',
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"{selected_dataset} a lo largo del tiempo",
            xaxis_title="Fecha",
            yaxis_title=f"{selected_dataset} ({unit})",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histograma":
        st.subheader(f"📊 Distribución de Valores - {selected_dataset}")
        
        fig = px.histogram(
            df_filtered,
            x='valor',
            nbins=30,
            title=f"Distribución de {selected_dataset}",
            labels={'valor': f'{selected_dataset} ({unit})', 'count': 'Frecuencia'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas descriptivas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Estadísticas Descriptivas")
            stats = df_filtered['valor'].describe()
            for stat, value in stats.items():
                st.write(f"**{stat.title()}:** {value:.3f}")
        
        with col2:
            st.subheader("📊 Percentiles")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                value = np.percentile(df_filtered['valor'], p)
                st.write(f"**Percentil {p}:** {value:.3f}")
    
    elif chart_type == "Box Plot":
        st.subheader(f"📦 Box Plot - {selected_dataset}")
        
        # Box plot por año
        df_filtered['año'] = df_filtered['fecha'].dt.year
        
        fig = px.box(
            df_filtered,
            x='año',
            y='valor',
            title=f"Distribución anual de {selected_dataset}",
            labels={'valor': f'{selected_dataset} ({unit})', 'año': 'Año'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
          # Box plot por mes
        df_filtered['mes'] = df_filtered['fecha'].dt.month
        month_names = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        df_filtered['mes_nombre'] = df_filtered['mes'].map(month_names)
        
        fig2 = px.box(
            df_filtered,
            x='mes_nombre',
            y='valor',
            title=f"Distribución mensual de {selected_dataset}",
            labels={'valor': f'{selected_dataset} ({unit})', 'mes_nombre': 'Mes'}
        )
        
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif chart_type == "Estadísticas":
        st.subheader(f"📈 Análisis Estadístico - {selected_dataset}")
        
        # Crear gráficos de estadísticas
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Promedio Mensual', 'Variación Anual', 'Tendencia', 'Completitud de Datos'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Promedio mensual
        monthly_avg = df_filtered.groupby(df_filtered['fecha'].dt.month)['valor'].mean()
        fig.add_trace(
            go.Bar(x=list(month_names.values()), y=monthly_avg.values, name='Promedio Mensual'),
            row=1, col=1
        )
        
        # Variación anual
        yearly_stats = df_filtered.groupby(df_filtered['fecha'].dt.year)['valor'].agg(['mean', 'std'])
        fig.add_trace(
            go.Scatter(x=yearly_stats.index, y=yearly_stats['mean'], name='Promedio Anual', mode='lines+markers'),
            row=1, col=2
        )
        
        # Tendencia (media móvil de 30 días)
        df_filtered['media_movil'] = df_filtered['valor'].rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(x=df_filtered['fecha'], y=df_filtered['media_movil'], name='Tendencia (30 días)', mode='lines'),
            row=2, col=1
        )
        
        # Completitud de datos
        monthly_completeness = df_filtered.groupby(df_filtered['fecha'].dt.month)['completo_mediciones'].mean()
        fig.add_trace(
            go.Scatter(x=list(month_names.values()), y=monthly_completeness.values, 
                      name='Completitud (%)', mode='lines+markers', fill='tonexty'),
            row=2, col=2        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Análisis de Tendencia":
        st.subheader(f"📈 Análisis de Tendencia - {selected_dataset}")
        
        # Calcular tendencia usando regresión lineal simple con numpy
        df_filtered['dias'] = (df_filtered['fecha'] - df_filtered['fecha'].min()).dt.days
        x = df_filtered['dias'].values
        y = df_filtered['valor'].values
        
        # Calcular coeficientes de regresión lineal
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n
        
        # Predicciones
        y_pred = slope * x + intercept
        df_filtered['tendencia'] = y_pred
        
        # Calcular R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Gráfico de tendencia
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['fecha'],
            y=df_filtered['valor'],
            mode='markers',
            name='Datos Observados',
            opacity=0.6,
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_filtered['fecha'],
            y=df_filtered['tendencia'],
            mode='lines',
            name='Línea de Tendencia',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title=f"Análisis de Tendencia - {selected_dataset}",
            xaxis_title="Fecha",
            yaxis_title=f"{selected_dataset} ({unit})",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de tendencia
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pendiente", f"{slope:.6f} {unit}/día")
        with col2:
            st.metric("R² (Ajuste)", f"{r_squared:.4f}")
        with col3:
            trend_direction = "📈 Ascendente" if slope > 0 else "📉 Descendente"
            st.metric("Dirección", trend_direction)
    
    elif chart_type == "Comparación Anual":
        st.subheader(f"📅 Comparación Anual - {selected_dataset}")
        
        # Gráfico de comparación anual
        df_filtered['año'] = df_filtered['fecha'].dt.year
        df_filtered['dia_año'] = df_filtered['fecha'].dt.dayofyear
        
        fig = go.Figure()
        
        for year in sorted(df_filtered['año'].unique()):
            year_data = df_filtered[df_filtered['año'] == year]
            fig.add_trace(go.Scatter(
                x=year_data['dia_año'],
                y=year_data['valor'],
                mode='lines',
                name=str(year),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"Comparación Anual - {selected_dataset}",
            xaxis_title="Día del Año",
            yaxis_title=f"{selected_dataset} ({unit})",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas anuales
        yearly_stats = df_filtered.groupby('año')['valor'].agg(['mean', 'max', 'min', 'std']).round(3)
        st.subheader("📊 Estadísticas Anuales")
        st.dataframe(yearly_stats, use_container_width=True)

# Panel de comparación múltiple
st.markdown("---")
st.subheader("🔄 Comparación Múltiple de Variables")

# Selector múltiple
selected_vars = st.multiselect(
    "Selecciona variables para comparar:",
    list(datasets.keys()),
    default=list(datasets.keys())[:3]
)

if len(selected_vars) > 1:
    # Normalizar datos para comparación
    fig = go.Figure()
    
    for var in selected_vars:
        df_var = datasets[var]['data']
        # Normalizar entre 0 y 1
        values_norm = (df_var['valor'] - df_var['valor'].min()) / (df_var['valor'].max() - df_var['valor'].min())
        
        fig.add_trace(go.Scatter(
            x=df_var['fecha'],
            y=values_norm,
            mode='lines',
            name=var,
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Comparación de Variables (Valores Normalizados 0-1)",
        xaxis_title="Fecha",
        yaxis_title="Valor Normalizado",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dashboard Meteorológico e Hidrológico - Estación Papallacta</p>
        <p>Datos procesados y visualizados con Streamlit 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

def run_main_dashboard():
    """Función principal del dashboard - se ejecuta desde main.py"""
    pass  # Todo el código ya está en el nivel superior

if __name__ == "__main__":
    # Solo se ejecuta cuando se llama directamente
    pass
