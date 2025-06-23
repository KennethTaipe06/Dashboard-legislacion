@echo off
echo ================================================
echo   Sistema de Análisis Meteorológico - Papallacta
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
python -c "import statsmodels" 2>nul
if errorlevel 1 (
    echo ADVERTENCIA: statsmodels no está instalado.
    echo Algunas funciones avanzadas de forecasting no estarán disponibles.
    echo Para instalar: pip install statsmodels
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

echo Iniciando el sistema...
echo El sistema se abrirá en tu navegador en: http://localhost:8503
echo.
echo Para detener el sistema, presiona Ctrl+C
echo.

streamlit run main.py --server.port 8503

pause
