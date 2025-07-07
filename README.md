# Sistema de Predicciones Meteorológicas e Hidrológicas - Papallacta

Este sistema especializado permite realizar predicciones (forecasting) de datos meteorológicos e hidrológicos de la estación Papallacta con múltiples algoritmos y validación automática.

## 🌟 Características Principales

### � Módulo de Predicciones (Forecasting)
- **Predicciones a largo plazo**: Hasta 2 años de forecast
- **Validación 80/20**: División automática para entrenamiento y prueba
- **Múltiples algoritmos**:
  - **Regresión Lineal Simple**: Predicciones basadas en tendencias lineales
  - **Naive Estacional**: Repite patrones del año anterior
  - **Promedio Móvil**: Suavizado con ventanas móviles
  - **Machine Learning con scikit-learn** (opcional):
    - Regresión Lineal Avanzada (con características cíclicas)
    - Regresión Polinomial (grados 2-5)
    - Forecasting Estacional ML
- **Métricas de precisión reales**: Evaluación en datos de prueba no vistos
- **Intervalos de confianza**: Cálculos estadísticos para todas las predicciones
- **Comparación de métodos**: Visualización simultánea de diferentes algoritmos
- **Métricas de evaluación completas**: MAE, RMSE, MAPE, R², Precisión (%)
- **Exportación de resultados**: Descarga en formato CSV
- **Interfaz intuitiva**: Controles fáciles de usar para cada método

## 📊 Variables Disponibles para Predicción

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
iniciar_sistema.bat
```

#### Opción 2: Ejecución Manual
```bash
streamlit run main.py
```

3. **Abrir en el navegador:**
   El sistema se abrirá automáticamente en `http://localhost:8503`

## 🎯 Navegación del Sistema

### Menú Principal
El archivo `main.py` actúa como hub central con dos opciones:

1. **� Predicciones (Forecasting)** (Página principal)
   - Módulo especializado en predicciones meteorológicas e hidrológicas
   - Múltiples algoritmos de forecasting
   - Validación automática y métricas de precisión
2. **📚 Información del Sistema**
   - Documentación completa del sistema de predicciones
   - Estadísticas del sistema
   - Guía de uso y métricas

## 🔮 Cómo Usar el Sistema de Predicciones

### Panel de Control (Sidebar)
- **Seleccionar Variable**: Elige la variable meteorológica o hidrológica para predecir
- **Período de Predicción**: Define cuántos días hacia el futuro predecir (30-730 días)
- **Métodos de Forecasting**: Selecciona qué algoritmos usar
- **Parámetros Avanzados**: Personaliza configuraciones específicas de cada método

### Métodos de Forecasting Disponibles

1. **Regresión Lineal**
   - Predicciones basadas en tendencias lineales simples
   - Rápido y eficiente para tendencias claras
   - Incluye intervalos de confianza

2. **Naive Estacional**
   - Repite los patrones del año anterior
   - Ideal para datos con fuerte estacionalidad
   - Simple pero efectivo

3. **Promedio Móvil**
   - Suavizado usando ventanas móviles
   - Configurable: tamaño de ventana
   - Bueno para reducir ruido

4. **Machine Learning (Opcional)**
   - **Regresión Lineal ML**: Con características cíclicas estacionales
   - **Regresión Polinomial**: Tendencias no lineales (grados 2-5)
   - **Estacional ML**: Combina ML con patrones estacionales
   - Requiere: `pip install scikit-learn`

### Métricas de Evaluación

- **MAE (Mean Absolute Error)**: Error absoluto promedio
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual promedio
- **R² (Coeficiente de Determinación)**: Calidad del ajuste (0-1)
- **Precisión (%)**: Porcentaje de precisión general

### Validación 80/20

El sistema automáticamente:
1. **Divide los datos**: 80% para entrenamiento, 20% para prueba
2. **Entrena modelos**: Solo usa datos de entrenamiento
3. **Evalúa precisión**: Prueba en datos no vistos
4. **Calcula métricas**: Métricas reales de precisión

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
