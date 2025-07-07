@echo off
echo ================================================
echo   Sistema de Predicciones Meteorológicas - Papallacta
echo ================================================
echo.

echo Verificando dependencias...
python -c "import streamlit, pandas, plotly, numpy" 2>nul
if errorlevel 1 (
    echo ERROR: Faltan dependencias básicas.
    echo Instalando dependencias...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: No se pudieron instalar las dependencias.
        echo Por favor, ejecuta manualmente: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo Dependencias básicas verificadas ✓
echo.

echo Verificando dependencias avanzadas (opcional)...
python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo ADVERTENCIA: scikit-learn no está instalado.
    echo Algunas funciones avanzadas de ML no estarán disponibles.
    echo Para instalar: pip install scikit-learn
    echo.
)

python -c "import folium, streamlit_folium" 2>nul
if errorlevel 1 (
    echo ADVERTENCIA: Folium no está instalado.
    echo Los mapas interactivos no estarán disponibles.
    echo Para instalar: pip install folium streamlit-folium
    echo.
)

echo Verificando estructura de datos...
if not exist "datasets_limpios" (
    echo ERROR: Carpeta 'datasets_limpios' no encontrada.
    echo Asegúrate de que los archivos de datos estén en la carpeta correcta.
    pause
    exit /b 1
)

echo Estructura de datos verificada ✓
echo.

echo Iniciando el Sistema de Predicciones...
echo El sistema se abrirá en tu navegador en: http://localhost:8503
echo.
echo 🔮 SISTEMA DE FORECASTING Y MAPAS METEOROLÓGICOS 🔮
echo - Predicciones de variables meteorológicas e hidrológicas
echo - Múltiples algoritmos de forecasting disponibles
echo - Validación automática con métricas de precisión
echo - Intervalos de confianza estadísticos
echo - Mapas interactivos con Folium
echo - Visualización geoespacial de datos meteorológicos
echo.
echo Para detener el sistema, presiona Ctrl+C
echo.

streamlit run main.py --server.port 8503

pause
