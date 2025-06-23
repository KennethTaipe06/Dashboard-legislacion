@echo off
echo ================================================
echo  Sistema de Analisis Meteorologico - Papallacta
echo ================================================
echo.
echo Selecciona una opcion:
echo.
echo 1. Dashboard Principal (Visualizaciones)
echo 2. Sistema Completo (Dashboard + Forecasting)
echo 3. Solo Modulo de Forecasting
echo 4. Salir
echo.
set /p option="Ingresa el numero de tu opcion: "

if "%option%"=="1" (
    echo.
    echo Iniciando Dashboard Principal...
    streamlit run dashboard.py --server.port 8501
) else if "%option%"=="2" (
    echo.
    echo Iniciando Sistema Completo...
    streamlit run main.py --server.port 8502
) else if "%option%"=="3" (
    echo.
    echo Iniciando Modulo de Forecasting...
    streamlit run forecast_module.py --server.port 8503
) else if "%option%"=="4" (
    echo.
    echo Saliendo...
    exit /b 0
) else (
    echo.
    echo Opcion invalida. Intenta de nuevo.
    pause
    goto :eof
)

pause
