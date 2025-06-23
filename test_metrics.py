#!/usr/bin/env python3
"""
Script de prueba para verificar las métricas de precisión
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forecast_module import simple_linear_forecast, calculate_forecast_metrics, prepare_forecast_data

def test_metrics():
    """Prueba las métricas de precisión con datos sintéticos"""
    print("🧪 Probando métricas de precisión con validación 80/20...")
    
    # Crear datos sintéticos con tendencia y ruido
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generar serie temporal sintética
    trend = np.linspace(10, 20, n)  # Tendencia lineal
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Componente estacional
    noise = np.random.normal(0, 0.5, n)  # Ruido aleatorio
    
    values = trend + seasonal + noise
    
    # Crear DataFrame
    df = pd.DataFrame({
        'fecha': dates,
        'valor': values
    })
    
    print(f"📊 Datos generados: {len(df)} observaciones")
    print(f"📅 Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    
    # Preparar serie temporal
    series = prepare_forecast_data(df)
    
    # Probar regresión lineal con validación 80/20
    forecast_df, metrics = simple_linear_forecast(series, periods=365)
    
    print("\n🎯 Métricas de precisión (validación 80/20):")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R²']:.4f}")
    print(f"  Precisión: {metrics['Precisión (%)']:.1f}%")
    print(f"  Correlación: {metrics['Correlación']:.4f}")
    print(f"  Accuracy ±5%: {metrics['Accuracy ±5%']:.1f}%")
    print(f"  Accuracy ±10%: {metrics['Accuracy ±10%']:.1f}%")
    print(f"  Accuracy ±20%: {metrics['Accuracy ±20%']:.1f}%")
    
    # Verificar que las métricas están en rangos razonables
    assert 0 <= metrics['Precisión (%)'] <= 100, "Precisión fuera de rango"
    assert -1 <= metrics['Correlación'] <= 1, "Correlación fuera de rango"
    assert 0 <= metrics['R²'] <= 1, "R² fuera de rango esperado"
    
    print("\n✅ Todas las métricas están en rangos válidos!")
    print(f"📈 Predicciones generadas: {len(forecast_df)} días futuros")
    
    return True

if __name__ == "__main__":
    try:
        test_metrics()
        print("\n🎉 Prueba completada exitosamente!")
    except Exception as e:
        print(f"\n❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()
