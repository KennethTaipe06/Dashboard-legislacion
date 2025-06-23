import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis Meteorológico - Papallacta",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Título principal
    st.title("🌦️ Sistema de Análisis Meteorológico e Hidrológico")
    st.markdown("### Estación Papallacta")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("🧭 Navegación")
    
    # Opciones del menú
    menu_options = {
        "📊 Dashboard Principal": "dashboard",
        "🔮 Predicciones (Forecasting)": "forecasting",
        "📚 Información del Sistema": "info"
    }
    
    selected_option = st.sidebar.selectbox(
        "Selecciona una opción:",
        list(menu_options.keys())
    )
    
    # Ejecutar la opción seleccionada
    option_key = menu_options[selected_option]
    
    if option_key == "dashboard":
        # Importar y ejecutar dashboard principal
        try:
            import dashboard
            # El dashboard se ejecutará automáticamente al importarlo
        except ImportError as e:
            st.error(f"Error al cargar el dashboard principal: {e}")
        except Exception as e:
            st.error(f"Error inesperado en el dashboard: {e}")
    
    elif option_key == "forecasting":
        # Importar y ejecutar módulo de forecasting
        try:
            from forecast_module import run_forecast_module
            run_forecast_module()
        except ImportError as e:
            st.error(f"Error al cargar el módulo de forecasting: {e}")
            st.info("Asegúrate de que todas las dependencias estén instaladas: `pip install -r requirements.txt`")
        except Exception as e:
            st.error(f"Error inesperado en forecasting: {e}")
    
    elif option_key == "info":
        show_system_info()

def show_system_info():
    """Muestra información del sistema"""
    st.header("📚 Información del Sistema")
    
    st.markdown("""
    ## 🌟 Descripción General
    
    Este sistema proporciona herramientas completas para el análisis de datos meteorológicos e 
    hidrológicos de la estación Papallacta, incluyendo:
    
    ### 📊 Dashboard Principal
    - **Visualización interactiva** de series temporales
    - **Análisis estadístico** completo con histogramas y box plots
    - **Comparación temporal** entre diferentes períodos
    - **Métricas en tiempo real** de las variables
    - **Filtros personalizables** por fecha y variable
    
    ### 🔮 Módulo de Predicciones
    - **Forecasting a corto y largo plazo** (hasta 2 años)
    - **Múltiples algoritmos** de predicción:
      - Regresión Lineal
      - Naive Estacional
      - Promedio Móvil
      - Suavizado Exponencial (Holt-Winters)
      - ARIMA
    - **Descomposición de series temporales**
    - **Intervalos de confianza** para todas las predicciones
    - **Comparación de métodos** en un solo gráfico
    - **Exportación de resultados** en formato CSV
    
    ## 📈 Variables Disponibles
    
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
    - **SciPy**: Análisis estadístico
    - **Statsmodels**: Modelos estadísticos y forecasting avanzado
    - **Scikit-learn**: Machine learning (opcional)
    
    ## 🚀 Características Avanzadas
    
    ### Dashboard
    - **Caching inteligente** para mejor rendimiento
    - **Responsive design** adaptable a diferentes pantallas
    - **Tooltips interactivos** con información detallada
    - **Zoom y pan** en todos los gráficos
    - **Métricas comparativas** con períodos anteriores
    
    ### Forecasting
    - **Validación automática** de la calidad de los datos
    - **Detección de outliers** y limpieza de datos
    - **Análisis de estacionalidad** automático
    - **Métricas de evaluación** (MAE, RMSE, MAPE)
    - **Intervalos de confianza** calculados estadísticamente
    
    ## 📊 Casos de Uso
    
    1. **Monitoreo en tiempo real** de condiciones meteorológicas
    2. **Análisis de tendencias** climáticas a largo plazo
    3. **Predicción de eventos** meteorológicos extremos
    4. **Planificación agrícola** basada en datos históricos
    5. **Gestión de recursos hídricos**
    6. **Investigación climatológica**
    
    ## 🔧 Instalación y Configuración
    
    ```bash
    # Instalar dependencias
    pip install -r requirements.txt
    
    # Ejecutar el sistema
    streamlit run main.py
    ```
    
    ## 📞 Soporte Técnico
    
    Para problemas técnicos:
    1. Verifica que todos los archivos CSV estén en `datasets_limpios/`
    2. Asegúrate de que las dependencias estén instaladas
    3. Revisa los logs en la consola para errores específicos
    
    ---
    
    **Desarrollado para el análisis avanzado de datos meteorológicos e hidrológicos** 🌦️📊
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
