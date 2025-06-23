# Sistema de Análisis Meteorológico e Hidrológico - Papallacta

Este sistema integral permite visualizar, analizar y predecir datos meteorológicos e hidrológicos de la estación Papallacta de manera intuitiva y completa.

## 🌟 Características Principales

### 📊 Dashboard Principal
- **Visualización de Series Temporales**: Gráficos de líneas interactivos con valores máximos y mínimos absolutos
- **Análisis Estadístico**: Histogramas, box plots y estadísticas descriptivas
- **Análisis de Tendencias**: Regresión lineal y métricas de ajuste
- **Comparación Anual y Mensual**: Visualización de patrones estacionales
- **Comparación Múltiple**: Análisis simultáneo de múltiples variables
- **Filtros Interactivos**: Selección de rangos de fechas y variables
- **Métricas en Tiempo Real**: KPIs principales actualizados dinámicamente

### 🔮 Módulo de Predicciones (Forecasting)
- **Predicciones a largo plazo**: Hasta 2 años de forecast
- **Múltiples algoritmos**:
  - Regresión Lineal Simple
  - Naive Estacional (repite patrones del año anterior)
  - Promedio Móvil
  - **Machine Learning con scikit-learn**:
    - Regresión Lineal Avanzada (con características cíclicas)
    - Regresión Polinomial (grados 2-5)
    - Forecasting Estacional ML
  - **Métodos Estadísticos Avanzados** (con statsmodels):
    - Suavizado Exponencial
    - ARIMA
  - Promedio Móvil con tendencia
  - Suavizado Exponencial (Holt-Winters)
  - Modelos ARIMA
- **Descomposición de series temporales**: Análisis de tendencia, estacionalidad y residuos
- **Intervalos de confianza**: Para todas las predicciones
- **Comparación de métodos**: Visualización simultánea de diferentes algoritmos
- **Métricas de evaluación**: MAE, RMSE, MAPE
- **Exportación de resultados**: Descarga en formato CSV

## 📊 Variables Disponibles

- **Hidrológicas:**
  - Caudal (m³/s)
  - Nivel de Agua (m)

- **Meteorológicas:**
  - Temperatura Ambiente (°C)
  - Precipitación (mm)
  - Humedad Relativa (%)
  - Presión Atmosférica (hPa)
  - Radiación Solar (W/m²)
  - Velocidad del Viento (m/s)
  - Dirección del Viento (°)

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación

#### Opción 1: Instalación Automática (Windows)
```bash
# Ejecutar el instalador automático
install.bat
```

#### Opción 2: Instalación Manual
1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar instalación:**
   ```bash
   python -c "import streamlit, pandas, plotly, numpy, sklearn, scipy, statsmodels"
   ```

### Ejecución

#### Opción 1: Script de Inicio Rápido (Windows)
```bash
run.bat
```

#### Opción 2: Ejecución Manual
1. **Sistema completo:**
   ```bash
   streamlit run main.py
   ```

2. **Módulos individuales:**
   ```bash
   # Solo dashboard principal
   streamlit run dashboard.py
   
   # Solo módulo de forecasting
   streamlit run forecast_module.py
   ```

4. **Abrir en el navegador:**
   El sistema se abrirá automáticamente en `http://localhost:8501`

## 🎯 Navegación del Sistema

### Menú Principal
El archivo `main.py` actúa como hub central con tres opciones:

1. **📊 Dashboard Principal**
   - Visualización y análisis de datos históricos
   - Todos los tipos de gráficos y análisis estadísticos

2. **🔮 Predicciones (Forecasting)**
   - Módulo especializado en predicciones
   - Múltiples algoritmos de forecasting
   - Descomposición de series temporales

3. **📚 Información del Sistema**
   - Documentación completa
   - Estadísticas del sistema
   - Guía de uso

## 🎯 Cómo Usar el Dashboard

### Panel de Control (Sidebar)
- **Seleccionar Variable**: Elige la variable meteorológica o hidrológica a visualizar
- **Filtro de Fechas**: Define el rango temporal de análisis
- **Tipo de Gráfico**: Selecciona entre diferentes tipos de visualización

### Tipos de Visualización

1. **Serie Temporal**
   - Visualización cronológica de los datos
   - Incluye valores máximos y mínimos absolutos
   - Interactividad con zoom y tooltips

2. **Histograma**
   - Distribución de frecuencias de los valores
   - Estadísticas descriptivas completas
   - Análisis de percentiles

3. **Box Plot**
   - Distribución por años y meses
   - Identificación de outliers
   - Análisis de variabilidad estacional

4. **Estadísticas**
   - Promedio mensual
   - Variación anual
   - Tendencia (media móvil 30 días)
   - Completitud de datos

5. **Análisis de Tendencia**
   - Regresión lineal
   - Coeficiente de determinación (R²)
   - Dirección de la tendencia

### Módulo de Forecasting

1. **Configuración de Predicción**
   - Selecciona la variable a predecir
   - Define el período de predicción (30-730 días)
   - Elige los métodos de forecasting a usar

2. **Métodos Disponibles**
   - **Regresión Lineal**: Tendencia lineal simple
   - **Naive Estacional**: Repite patrones del año anterior
   - **Promedio Móvil**: Media móvil con tendencia
   - **Suavizado Exponencial**: Holt-Winters (requiere statsmodels)
   - **ARIMA**: Modelo autoregresivo (requiere statsmodels)

3. **Análisis de Resultados**
   - Gráfico comparativo de todos los métodos
   - Intervalos de confianza del 95%
   - Estadísticas de las predicciones
   - Métricas de evaluación

4. **Descomposición de Series**
   - Tendencia de largo plazo
   - Componente estacional
   - Residuos aleatorios

5. **Exportación**
   - Descarga de predicciones en CSV
   - Incluye todos los métodos y sus intervalos de confianza

### Métricas Principales
En la parte superior se muestran:
- Valor promedio del período seleccionado
- Valor máximo registrado
- Valor mínimo registrado
- Porcentaje de completitud de datos

### Comparación Múltiple
- Selecciona múltiples variables para comparar
- Visualización normalizada (0-1) para facilitar la comparación
- Identificación de correlaciones visuales

## 📁 Estructura del Proyecto

```
Dashboard legislacion/
├── main.py                 # Archivo principal (menú de navegación)
├── dashboard.py            # Dashboard principal de visualización
├── forecast_module.py      # Módulo de predicciones
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Documentación
├── datasets_limpios/      # Datos procesados
│   ├── H34-Papallacta_Caudal-Diario.csv
│   ├── H34-Papallacta_Nivel_de_agua-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Temperatura_ambiente-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Precipitación-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Humedad_relativa-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Presion_atmosférica-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Radiación_solar-Diario.csv
│   ├── M5025-La_Virgen_Papallacta_Velocidad_de_viento-Diario.csv
│   └── M5025-La_Virgen_Papallacta_Dirección_de_viento-Diario.csv
└── datasets/              # Datos originales (opcional)
```

## 🛠️ Dependencias

### Básicas (siempre requeridas)
- **Streamlit**: Framework para aplicaciones web
- **Pandas**: Manipulación y análisis de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Cálculos numéricos
- **SciPy**: Análisis estadístico

### Avanzadas (para forecasting completo)
- **Statsmodels**: Modelos estadísticos y forecasting
- **Scikit-learn**: Machine learning (opcional)

Si no tienes las dependencias avanzadas, el módulo de forecasting funcionará con métodos básicos (Regresión Lineal, Naive Estacional, Promedio Móvil).

### Formato de Datos

Los datasets deben estar en la carpeta `datasets_limpios/` con el siguiente formato:
- `fecha`: Fecha en formato YYYY/MM/DD
- `valor`: Valor de la medición
- `max_abs`: Valor máximo absoluto (opcional)
- `min_abs`: Valor mínimo absoluto (opcional)
- `completo_mediciones`: Porcentaje de completitud
- `completo_umbral`: Porcentaje de completitud según umbral

## 🎨 Características Técnicas

### Dashboard Principal
- **Caching inteligente**: Los datos se cargan una sola vez para mejorar el rendimiento
- **Responsive Design**: Adaptado para diferentes tamaños de pantalla
- **Visualizaciones Interactivas**: Zoom, pan, tooltips y selecciones
- **Exportación**: Posibilidad de descargar gráficos como imágenes

### Módulo de Forecasting
- **Validación automática**: Limpieza y validación de datos
- **Detección de outliers**: Remoción automática de valores extremos
- **Análisis de estacionalidad**: Detección automática de patrones estacionales  
- **Métricas de evaluación**: MAE, RMSE, MAPE para cada método
- **Intervalos de confianza**: Calculados estadísticamente para cada predicción

## 🎨 Personalización

El dashboard es completamente personalizable:
- Colores y temas
- Tipos de gráficos adicionales
- Métricas personalizadas
- Filtros adicionales

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError"
Verifica que la carpeta `datasets_limpios/` exista y contenga los archivos CSV.

### Puerto ocupado
```bash
streamlit run dashboard.py --server.port 8502
```

## 📞 Soporte

Para preguntas o problemas técnicos, revisa:
1. Los logs en la consola donde ejecutaste Streamlit
2. La documentación de Streamlit: https://docs.streamlit.io/
3. La documentación de Plotly: https://plotly.com/python/

---
*Dashboard desarrollado para el análisis de datos meteorológicos e hidrológicos de la estación Papallacta* 🌦️📊
