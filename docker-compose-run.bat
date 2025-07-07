@echo off
echo ================================================
echo   Sistema de Predicciones Meteorológicas - Docker Compose
echo ================================================
echo.

echo Verificando si Docker Compose está instalado...
docker compose version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose no está instalado.
    echo Por favor, instala Docker Desktop que incluye Docker Compose.
    pause
    exit /b 1
)

echo Docker Compose detectado ✓
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

echo Iniciando el sistema con Docker Compose...
echo El sistema se abrirá en tu navegador en: http://localhost:8503
echo.
echo 🔮 SISTEMA DE FORECASTING Y MAPAS METEOROLÓGICOS (DOCKER COMPOSE) 🔮
echo - Predicciones de variables meteorológicas e hidrológicas
echo - Múltiples algoritmos de forecasting disponibles
echo - Validación automática con métricas de precisión
echo - Intervalos de confianza estadísticos
echo - Mapas interactivos con Folium
echo - Visualización geoespacial de datos meteorológicos
echo.
echo Para detener el sistema, ejecuta: docker compose down
echo.

docker compose up --build

pause
