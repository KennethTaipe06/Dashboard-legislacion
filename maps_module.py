import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# Importaciones opcionales
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

@st.cache_data
def load_datasets():
    """Carga todos los datasets desde la carpeta datasets_limpios"""
    datasets = {}
    datasets_path = "datasets_limpios"
    
    # Lista de archivos en la carpeta
    files = [
        "H34-Papallacta_Caudal-Diario.csv",
        "H34-Papallacta_Nivel_de_agua-Diario.csv",
        "M5025-La_Virgen_Papallacta_Dirección_de_viento-Diario.csv",
        "M5025-La_Virgen_Papallacta_Humedad_relativa-Diario.csv",
        "M5025-La_Virgen_Papallacta_Precipitación-Diario.csv",
        "M5025-La_Virgen_Papallacta_Presion_atmosférica-Diario.csv",
        "M5025-La_Virgen_Papallacta_Radiación_solar-Diario.csv",
        "M5025-La_Virgen_Papallacta_Temperatura_ambiente-Diario.csv",
        "M5025-La_Virgen_Papallacta_Velocidad_de_viento-Diario.csv"
    ]
    
    # Mapeo de nombres amigables
    friendly_names = {
        "H34-Papallacta_Caudal-Diario.csv": "Caudal",
        "H34-Papallacta_Nivel_de_agua-Diario.csv": "Nivel de Agua",
        "M5025-La_Virgen_Papallacta_Dirección_de_viento-Diario.csv": "Dirección del Viento",
        "M5025-La_Virgen_Papallacta_Humedad_relativa-Diario.csv": "Humedad Relativa",
        "M5025-La_Virgen_Papallacta_Precipitación-Diario.csv": "Precipitación",
        "M5025-La_Virgen_Papallacta_Presion_atmosférica-Diario.csv": "Presión Atmosférica",
        "M5025-La_Virgen_Papallacta_Radiación_solar-Diario.csv": "Radiación Solar",
        "M5025-La_Virgen_Papallacta_Temperatura_ambiente-Diario.csv": "Temperatura Ambiente",
        "M5025-La_Virgen_Papallacta_Velocidad_de_viento-Diario.csv": "Velocidad del Viento"
    }
    
    # Unidades para cada variable
    units = {
        "Caudal": "m³/s",
        "Nivel de Agua": "m",
        "Dirección del Viento": "°",
        "Humedad Relativa": "%",
        "Precipitación": "mm",
        "Presión Atmosférica": "hPa",
        "Radiación Solar": "W/m²",
        "Temperatura Ambiente": "°C",
        "Velocidad del Viento": "m/s"
    }
    
    for file in files:
        try:
            df = pd.read_csv(os.path.join(datasets_path, file))
            df['fecha'] = pd.to_datetime(df['fecha'])
            df = df.sort_values('fecha')
            
            name = friendly_names[file]
            datasets[name] = {
                'data': df,
                'unit': units[name]
            }
        except Exception as e:
            st.warning(f"Error cargando {file}: {e}")
    
    return datasets

def show_maps_interface():
    """Interfaz principal para mostrar mapas"""
    
    if not FOLIUM_AVAILABLE:
        st.error("📍 Folium no está disponible. Instala con: `pip install folium streamlit-folium`")
        return
    
    st.title("🗺️ Visualización de Mapas Meteorológicos")
    st.markdown("---")
    
    # Cargar datasets
    datasets = load_datasets()
    
    if not datasets:
        st.error("❌ No se pudieron cargar los datos. Verifica que la carpeta 'datasets_limpios' existe y contiene los archivos CSV.")
        return
    
    # Sidebar para configuración de mapas
    st.sidebar.header("🎛️ Configuración de Mapas")
    
    # Tipo de mapa
    map_type = st.sidebar.selectbox(
        "Tipo de Mapa:",
        ["Mapa Base", "Mapa de Calor", "Mapa de Marcadores", "Mapa Temático"],
        index=0
    )
    
    # Estilo del mapa
    map_style = st.sidebar.selectbox(
        "Estilo del Mapa:",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter", "Stamen Terrain", "Stamen Watercolor"],
        index=0
    )
    
    # Configuración de zoom
    zoom_level = st.sidebar.slider("Nivel de Zoom:", 1, 18, 6)
    
    # Mostrar el mapa seleccionado
    if map_type == "Mapa Base":
        show_base_map(map_style, zoom_level)
    elif map_type == "Mapa de Calor":
        show_heatmap(datasets, map_style, zoom_level)
    elif map_type == "Mapa de Marcadores":
        show_marker_map(datasets, map_style, zoom_level)
    elif map_type == "Mapa Temático":
        show_thematic_map(datasets, map_style, zoom_level)

def show_base_map(style="OpenStreetMap", zoom=6):
    """Muestra un mapa base simple"""
    st.subheader("🗺️ Mapa Base")
    
    # Coordenadas centrales de Ecuador
    ecuador_center = [-1.8312, -78.1834]
    
    # Crear mapa base
    m = folium.Map(
        location=ecuador_center,
        zoom_start=zoom,
        tiles=get_tile_style(style)
    )
    
    # Agregar algunos marcadores de ejemplo en Ecuador
    locations = [
        {"name": "Quito", "coords": [-0.1807, -78.4678], "info": "Capital de Ecuador"},
        {"name": "Guayaquil", "coords": [-2.1709, -79.9224], "info": "Puerto Principal"},
        {"name": "Cuenca", "coords": [-2.9001, -79.0059], "info": "Patrimonio Cultural"},
        {"name": "Papallacta", "coords": [-0.3667, -78.1500], "info": "Zona de Estudio - Estación Meteorológica"}
    ]
    
    for loc in locations:
        color = "red" if "Papallacta" in loc["name"] else "blue"
        folium.Marker(
            location=loc["coords"],
            popup=f"<b>{loc['name']}</b><br>{loc['info']}",
            tooltip=loc["name"],
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)
    
    # Mostrar el mapa
    map_data = st_folium(m, width=700, height=500)
    
    # Información del mapa
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎯 **Papallacta** (marcador rojo) es la ubicación de nuestras estaciones meteorológicas")
    with col2:
        if map_data['last_clicked']:
            lat = map_data['last_clicked']['lat']
            lng = map_data['last_clicked']['lng']
            st.success(f"📍 Último clic: {lat:.4f}, {lng:.4f}")

def show_heatmap(datasets, style="OpenStreetMap", zoom=6):
    """Muestra un mapa de calor con datos simulados"""
    st.subheader("🔥 Mapa de Calor")
    
    # Selector de variable para el mapa de calor
    selected_var = st.selectbox(
        "Variable para el Mapa de Calor:",
        list(datasets.keys()),
        index=0
    )
    
    ecuador_center = [-1.8312, -78.1834]
    
    # Crear mapa
    m = folium.Map(
        location=ecuador_center,
        zoom_start=zoom,
        tiles=get_tile_style(style)
    )
    
    # Generar datos simulados para el mapa de calor
    heat_data = generate_heat_data(selected_var)
    
    # Agregar capa de calor
    try:
        from folium.plugins import HeatMap
        HeatMap(heat_data).add_to(m)
    except ImportError:
        st.warning("Plugin HeatMap no disponible. Mostrando marcadores en su lugar.")
        # Alternativa: mostrar como marcadores coloreados
        for point in heat_data:
            lat, lng, intensity = point
            color = get_color_by_intensity(intensity)
            folium.CircleMarker(
                location=[lat, lng],
                radius=intensity * 10,
                popup=f"Intensidad: {intensity:.2f}",
                color=color,
                fill=True
            ).add_to(m)
    
    # Mostrar el mapa
    st_folium(m, width=700, height=500)
    
    st.info(f"🌡️ Mapa de calor simulado para: **{selected_var}**")
    st.caption("Los datos de calor son simulados para demostración. En un escenario real, se usarían datos de múltiples estaciones.")

def show_marker_map(datasets, style="OpenStreetMap", zoom=6):
    """Muestra un mapa con marcadores de estaciones"""
    st.subheader("📍 Mapa de Estaciones Meteorológicas")
    
    ecuador_center = [-1.8312, -78.1834]
    
    # Crear mapa
    m = folium.Map(
        location=ecuador_center,
        zoom_start=zoom,
        tiles=get_tile_style(style)
    )
    
    # Datos simulados de estaciones meteorológicas
    stations = generate_weather_stations()
    
    # Variable seleccionada para mostrar en los marcadores
    selected_var = st.selectbox(
        "Variable a mostrar en marcadores:",
        list(datasets.keys()),
        key="marker_var"
    )
    
    # Agregar marcadores de estaciones
    for station in stations:
        # Simular valor actual de la variable
        current_value = simulate_current_value(selected_var)
        unit = datasets[selected_var]['unit']
        
        # Color basado en el tipo de estación
        color = get_station_color(station['type'])
        
        popup_html = f"""
        <div style="width: 200px;">
            <h4>{station['name']}</h4>
            <p><b>Tipo:</b> {station['type']}</p>
            <p><b>Altitud:</b> {station['altitude']} m</p>
            <p><b>{selected_var}:</b> {current_value:.2f} {unit}</p>
            <p><b>Estado:</b> {station['status']}</p>
        </div>
        """
        
        folium.Marker(
            location=station['coords'],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{station['name']} - {current_value:.1f} {unit}",
            icon=folium.Icon(color=color, icon="thermometer")
        ).add_to(m)
    
    # Mostrar el mapa
    st_folium(m, width=700, height=500)
    
    # Leyenda
    st.markdown("### 📋 Leyenda de Estaciones")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Azul**: Meteorológica")
    with col2:
        st.markdown("🟢 **Verde**: Hidrológica")  
    with col3:
        st.markdown("🔴 **Rojo**: Principal (Papallacta)")

def show_thematic_map(datasets, style="OpenStreetMap", zoom=6):
    """Muestra un mapa temático con capas"""
    st.subheader("🎨 Mapa Temático")
    
    ecuador_center = [-1.8312, -78.1834]
    
    # Crear mapa base
    m = folium.Map(
        location=ecuador_center,
        zoom_start=zoom,
        tiles=get_tile_style(style)
    )
    
    # Capas temáticas
    st.markdown("### 🗂️ Capas Disponibles")
    col1, col2 = st.columns(2)
    
    with col1:
        show_precipitation = st.checkbox("💧 Precipitación", value=True)
        show_temperature = st.checkbox("🌡️ Temperatura", value=False)
    
    with col2:
        show_wind = st.checkbox("💨 Viento", value=False)
        show_topography = st.checkbox("⛰️ Topografía", value=True)
    
    # Agregar capas según selección
    if show_topography:
        add_topography_layer(m)
    
    if show_precipitation:
        add_precipitation_layer(m)
    
    if show_temperature:
        add_temperature_layer(m)
    
    if show_wind:
        add_wind_layer(m)
    
    # Control de capas
    folium.LayerControl().add_to(m)
    
    # Mostrar el mapa
    st_folium(m, width=700, height=500)
    
    st.info("🎯 Usa el control de capas en la esquina superior derecha del mapa para activar/desactivar capas")

# Funciones auxiliares

def get_tile_style(style):
    """Convierte el nombre del estilo a tiles de folium"""
    style_map = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB Positron": "CartoDB positron",
        "CartoDB Dark_Matter": "CartoDB dark_matter",
        "Stamen Terrain": "Stamen Terrain",
        "Stamen Watercolor": "Stamen Watercolor"
    }
    return style_map.get(style, "OpenStreetMap")

def generate_heat_data(variable):
    """Genera datos simulados para mapa de calor"""
    np.random.seed(42)  # Para resultados consistentes
    
    # Área aproximada de Ecuador
    lat_range = (-4.5, 1.5)
    lng_range = (-81.0, -75.0)
    
    heat_data = []
    for _ in range(100):  # 100 puntos de calor
        lat = np.random.uniform(*lat_range)
        lng = np.random.uniform(*lng_range)
        intensity = np.random.uniform(0.1, 1.0)
        heat_data.append([lat, lng, intensity])
    
    return heat_data

def get_color_by_intensity(intensity):
    """Retorna color basado en intensidad"""
    if intensity > 0.7:
        return "red"
    elif intensity > 0.4:
        return "orange"
    else:
        return "yellow"

def generate_weather_stations():
    """Genera datos simulados de estaciones meteorológicas"""
    stations = [
        {
            "name": "Papallacta Principal",
            "coords": [-0.3667, -78.1500],
            "type": "Meteorológica e Hidrológica",
            "altitude": 3300,
            "status": "Activa"
        },
        {
            "name": "Antisana",
            "coords": [-0.4833, -78.1167],
            "type": "Meteorológica",
            "altitude": 4200,
            "status": "Activa"
        },
        {
            "name": "Cotopaxi",
            "coords": [-0.6833, -78.4333],
            "type": "Meteorológica",
            "altitude": 3800,
            "status": "Mantenimiento"
        },
        {
            "name": "Río Papallacta",
            "coords": [-0.3500, -78.1400],
            "type": "Hidrológica",
            "altitude": 3250,
            "status": "Activa"
        },
        {
            "name": "Termas de Papallacta",
            "coords": [-0.3800, -78.1600],
            "type": "Meteorológica",
            "altitude": 3200,
            "status": "Activa"
        }
    ]
    return stations

def simulate_current_value(variable):
    """Simula un valor actual para una variable"""
    # Rangos típicos para cada variable
    ranges = {
        "Temperatura Ambiente": (5, 25),
        "Humedad Relativa": (40, 95),
        "Precipitación": (0, 50),
        "Presión Atmosférica": (650, 750),
        "Velocidad del Viento": (0, 15),
        "Dirección del Viento": (0, 360),
        "Radiación Solar": (0, 1000),
        "Caudal": (0.5, 10),
        "Nivel de Agua": (0.5, 3)
    }
    
    min_val, max_val = ranges.get(variable, (0, 100))
    return np.random.uniform(min_val, max_val)

def get_station_color(station_type):
    """Retorna color basado en el tipo de estación"""
    if "Principal" in station_type:
        return "red"
    elif "Hidrológica" in station_type:
        return "green"
    else:
        return "blue"

def add_topography_layer(map_obj):
    """Agrega capa de topografía"""
    # Simulamos puntos de elevación
    elevations = [
        {"coords": [-0.3667, -78.1500], "elevation": 3300, "name": "Papallacta"},
        {"coords": [-0.4833, -78.1167], "elevation": 4200, "name": "Antisana"},
        {"coords": [-0.6833, -78.4333], "elevation": 3800, "name": "Cotopaxi"},
    ]
    
    topography_group = folium.FeatureGroup(name="⛰️ Topografía")
    
    for point in elevations:
        folium.CircleMarker(
            location=point["coords"],
            radius=8,
            popup=f"{point['name']}: {point['elevation']} m",
            color="brown",
            fill=True,
            fillColor="orange",
            fillOpacity=0.7
        ).add_to(topography_group)
    
    topography_group.add_to(map_obj)

def add_precipitation_layer(map_obj):
    """Agrega capa de precipitación"""
    precipitation_group = folium.FeatureGroup(name="💧 Precipitación")
    
    # Simulamos zonas de precipitación
    for _ in range(20):
        lat = np.random.uniform(-1, 0)
        lng = np.random.uniform(-78.5, -77.5)
        rainfall = np.random.uniform(0, 30)
        
        color = "blue" if rainfall > 15 else "lightblue"
        
        folium.CircleMarker(
            location=[lat, lng],
            radius=rainfall/2,
            popup=f"Precipitación: {rainfall:.1f} mm",
            color=color,
            fill=True,
            fillOpacity=0.5
        ).add_to(precipitation_group)
    
    precipitation_group.add_to(map_obj)

def add_temperature_layer(map_obj):
    """Agrega capa de temperatura"""
    temperature_group = folium.FeatureGroup(name="🌡️ Temperatura")
    
    # Simulamos isotermas
    for _ in range(15):
        lat = np.random.uniform(-1, 0)
        lng = np.random.uniform(-78.5, -77.5)
        temp = np.random.uniform(5, 25)
        
        color = "red" if temp > 15 else "orange" if temp > 10 else "yellow"
        
        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            popup=f"Temperatura: {temp:.1f}°C",
            color=color,
            fill=True,
            fillOpacity=0.6
        ).add_to(temperature_group)
    
    temperature_group.add_to(map_obj)

def add_wind_layer(map_obj):
    """Agrega capa de viento"""
    wind_group = folium.FeatureGroup(name="💨 Viento")
    
    # Simulamos vectores de viento
    for _ in range(10):
        lat = np.random.uniform(-1, 0)
        lng = np.random.uniform(-78.5, -77.5)
        speed = np.random.uniform(0, 15)
        direction = np.random.uniform(0, 360)
        
        folium.Marker(
            location=[lat, lng],
            popup=f"Viento: {speed:.1f} m/s, {direction:.0f}°",
            icon=folium.Icon(color="green", icon="arrow-up")
        ).add_to(wind_group)
    
    wind_group.add_to(map_obj)

def run_maps_module():
    """Función principal del módulo de mapas"""
    # Esta función se llamará desde main.py
    pass
