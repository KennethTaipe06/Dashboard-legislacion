@echo off
echo ================================================
echo   Sistema de Predicciones Meteorológicas - Docker
echo ================================================
echo.

echo Verificando si Docker está instalado...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker no está instalado o no está en el PATH.
    echo Por favor, instala Docker Desktop desde: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker detectado ✓
echo.

echo Verificando si Docker está ejecutándose...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker no está ejecutándose.
    echo Por favor, inicia Docker Desktop y vuelve a intentar.
    pause
    exit /b 1
)

echo Docker está ejecutándose ✓
echo.

echo Construyendo la imagen Docker...
docker build -t dashboard-meteorologico .
if errorlevel 1 (
    echo ERROR: No se pudo construir la imagen Docker.
    pause
    exit /b 1
)

echo Imagen construida exitosamente ✓
echo.

echo Iniciando el contenedor...
echo El sistema se abrirá en tu navegador en: http://localhost:8503
echo.
echo 🔮 SISTEMA DE FORECASTING Y MAPAS METEOROLÓGICOS (DOCKER) 🔮
echo - Predicciones de variables meteorológicas e hidrológicas
echo - Múltiples algoritmos de forecasting disponibles
echo - Validación automática con métricas de precisión
echo - Intervalos de confianza estadísticos
echo - Mapas interactivos con Folium
echo - Visualización geoespacial de datos meteorológicos
echo.
echo Para detener el sistema, presiona Ctrl+C
echo.

docker run -p 8503:8503 -v "%cd%\datasets:/app/datasets:ro" -v "%cd%\datasets_limpios:/app/datasets_limpios:ro" dashboard-meteorologico

pause
