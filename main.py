import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicciones Meteorológicas - Papallacta",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Título principal
    st.title("🔮 Sistema de Predicciones Meteorológicas e Hidrológicas")
    st.markdown("### Estación Papallacta - Módulo de Forecasting")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("🧭 Navegación")
    
    # Opciones del menú (forecasting, mapas e info)
    menu_options = {
        "🔮 Predicciones (Forecasting)": "forecasting",
        "🗺️ Mapas Meteorológicos": "maps",
        "📚 Información del Sistema": "info"
    }
    
    selected_option = st.sidebar.selectbox(
        "Selecciona una opción:",
        list(menu_options.keys()),
        index=0  # Por defecto, mostrar forecasting
    )
    
    # Ejecutar la opción seleccionada
    option_key = menu_options[selected_option]
    
    if option_key == "forecasting":
        # Importar y ejecutar módulo de forecasting
        try:
            from forecast_module import run_forecast_module
            run_forecast_module()
        except ImportError as e:
            st.error(f"Error al cargar el módulo de forecasting: {e}")
            st.info("Asegúrate de que todas las dependencias estén instaladas: `pip install -r requirements.txt`")
        except Exception as e:
            st.error(f"Error inesperado en forecasting: {e}")
    
    elif option_key == "maps":
        # Importar y ejecutar módulo de mapas
        try:
            from maps_module import show_maps_interface
            show_maps_interface()
        except ImportError as e:
            st.error(f"Error al cargar el módulo de mapas: {e}")
            st.info("Para usar mapas, instala: `pip install folium streamlit-folium`")
        except Exception as e:
            st.error(f"Error inesperado en mapas: {e}")
    
    elif option_key == "info":
        show_system_info()

def show_system_info():
    """Muestra información del sistema"""
    st.header("📚 Información del Sistema de Predicciones")
    
    st.markdown("""
    ## 🌟 Descripción General
    
    Este sistema proporciona herramientas especializadas para el **forecasting (predicción)** de datos 
    meteorológicos e hidrológicos de la estación Papallacta.
    
    ### 🔮 Módulo de Predicciones
    - **Forecasting a corto y largo plazo** (hasta 2 años)
    - **Múltiples algoritmos** de predicción:
      - **Regresión Lineal**: Predicciones basadas en tendencias lineales
      - **Naive Estacional**: Predicciones simples basadas en patrones estacionales
      - **Promedio Móvil**: Predicciones suavizadas usando promedios móviles
      - **Regresión Polinomial**: Predicciones con tendencias no lineales (opcional)
      - **ML Estacional**: Machine Learning con componentes estacionales (opcional)
    - **Validación 80/20**: Métricas reales de precisión usando datos de prueba
    - **Intervalos de confianza** para todas las predicciones
    - **Comparación de métodos** en un solo gráfico
    - **Exportación de resultados** en formato CSV
    - **Métricas de evaluación**: MAE, RMSE, MAPE, R²
    
    ### 🗺️ Módulo de Mapas
    - **Mapas Base**: Visualización geográfica con diferentes estilos
    - **Mapas de Calor**: Representación de intensidad de variables
    - **Mapas de Marcadores**: Ubicación y datos de estaciones meteorológicas
    - **Mapas Temáticos**: Capas superpuestas (precipitación, temperatura, topografía)
    - **Interactividad**: Zoom, clic, tooltips y popups informativos
    - **Múltiples estilos**: OpenStreetMap, CartoDB, Stamen
    - **Simulación de datos**: Demostración con datos meteorológicos simulados
    
    ## 📈 Variables Disponibles para Predicción
    
    ### Hidrológicas
    - **Caudal** (m³/s)
    - **Nivel de Agua** (m)
    
    ### Meteorológicas
    - **Temperatura Ambiente** (°C)
    - **Precipitación** (mm)
    - **Humedad Relativa** (%)
    - **Presión Atmosférica** (hPa)
    - **Radiación Solar** (W/m²)
    - **Velocidad del Viento** (m/s)
    - **Dirección del Viento** (°)
    
    ## 🛠️ Tecnologías Utilizadas
    
    - **Streamlit**: Framework para aplicaciones web interactivas
    - **Pandas**: Manipulación y análisis de datos
    - **Plotly**: Visualizaciones interactivas avanzadas
    - **NumPy**: Cálculos numéricos eficientes
    - **Scikit-learn**: Machine learning para predicciones avanzadas (opcional)
    
    ## 🚀 Características del Sistema
    
    ### Predicciones Inteligentes
    - **Validación automática** de la calidad de los datos
    - **División 80/20** para entrenamiento y validación
    - **Detección automática** de patrones estacionales
    - **Cálculo de métricas** de precisión en tiempo real
    - **Intervalos de confianza** estadísticamente fundamentados
    
    ### Interfaz de Usuario
    - **Selección intuitiva** de variables y períodos
    - **Controles personalizables** para cada método
    - **Visualizaciones interactivas** con zoom y pan
    - **Exportación fácil** de resultados
    - **Métricas comparativas** entre métodos
    
    ## 📊 Casos de Uso Específicos
    
    1. **Predicción de caudales** para gestión de recursos hídricos
    2. **Pronóstico de temperatura** para planificación agrícola
    3. **Predicción de precipitación** para alertas tempranas
    4. **Forecasting de variables múltiples** para análisis integral
    5. **Validación de modelos** con métricas robustas
    6. **Comparación de algoritmos** para selección óptima
    
    ## 🔧 Instalación y Uso
    
    ```bash
    # Instalar dependencias básicas
    pip install streamlit pandas plotly numpy
    
    # Instalar dependencias opcionales para ML
    pip install scikit-learn
    
    # Ejecutar el sistema
    streamlit run main.py
    ```
    
    ## � Métricas de Evaluación
    
    - **MAE (Mean Absolute Error)**: Error absoluto promedio
    - **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
    - **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio
    - **R² (Coeficiente de Determinación)**: Calidad del ajuste del modelo
    - **Precisión (%)**: Porcentaje de precisión general del modelo
    
    ---
    
    **Sistema especializado en predicciones meteorológicas e hidrológicas** 🔮📊
    """)
    
    # Mostrar estadísticas del sistema
    st.subheader("📊 Estadísticas del Sistema")
    
    try:
        import os
        datasets_path = "datasets_limpios"
        if os.path.exists(datasets_path):
            files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Datasets Disponibles", len(files))
            with col2:
                total_size = sum(os.path.getsize(os.path.join(datasets_path, f)) for f in files)
                st.metric("Tamaño Total", f"{total_size / (1024*1024):.1f} MB")
            with col3:
                import pandas as pd
                total_records = 0
                for file in files:
                    try:
                        df = pd.read_csv(os.path.join(datasets_path, file))
                        total_records += len(df)
                    except:
                        pass
                st.metric("Registros Totales", f"{total_records:,}")
        else:
            st.warning("Carpeta 'datasets_limpios' no encontrada")
    
    except Exception as e:
        st.error(f"Error calculando estadísticas: {e}")

if __name__ == "__main__":
    main()
