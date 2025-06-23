#!/bin/bash

echo "================================================"
echo "   Sistema de Análisis Meteorológico - Papallacta"
echo "================================================"
echo

echo "Verificando dependencias..."
python3 -c "import streamlit, pandas, plotly, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Faltan dependencias básicas."
    echo "Instalando dependencias..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudieron instalar las dependencias."
        echo "Por favor, ejecuta manualmente: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "Dependencias básicas verificadas ✓"
echo

echo "Verificando dependencias avanzadas (opcional)..."
python3 -c "import statsmodels" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ADVERTENCIA: statsmodels no está instalado."
    echo "Algunas funciones avanzadas de forecasting no estarán disponibles."
    echo "Para instalar: pip3 install statsmodels"
    echo
fi

echo "Verificando estructura de datos..."
if [ ! -d "datasets_limpios" ]; then
    echo "ERROR: Carpeta 'datasets_limpios' no encontrada."
    echo "Asegúrate de que los archivos de datos estén en la carpeta correcta."
    exit 1
fi

echo "Estructura de datos verificada ✓"
echo

echo "Iniciando el sistema..."
echo "El sistema se abrirá en tu navegador en: http://localhost:8503"
echo
echo "Para detener el sistema, presiona Ctrl+C"
echo

streamlit run main.py --server.port 8503
