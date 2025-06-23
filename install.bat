@echo off
echo ========================================
echo  Instalador del Sistema de Forecasting
echo  Dashboard Meteorologico - Papallacta
echo ========================================
echo.

echo Instalando dependencias...
pip install -r requirements.txt

echo.
echo Verificando instalacion...
python -c "import streamlit, pandas, plotly, numpy, sklearn, scipy, statsmodels; print('✅ Todas las dependencias instaladas correctamente')"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Error en la instalacion. Verifica los mensajes de error anteriores.
    echo.
    echo Intentando instalacion individual...
    pip install streamlit pandas plotly numpy
    pip install scikit-learn scipy statsmodels
    pause
    exit /b 1
)

echo.
echo ✅ Instalacion completada exitosamente!
echo.
echo Para ejecutar el sistema:
echo   1. Dashboard Principal: streamlit run dashboard.py
echo   2. Sistema Completo: streamlit run main.py
echo   3. Solo Forecasting: streamlit run forecast_module.py
echo.
pause
