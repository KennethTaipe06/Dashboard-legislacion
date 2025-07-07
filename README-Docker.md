# 🐳 Sistema de Predicciones Meteorológicas - Docker

Este proyecto ha sido dockerizado para facilitar su despliegue y distribución. Ahora puedes ejecutar la aplicación en cualquier entorno que tenga Docker instalado.

## 📋 Prerrequisitos

- **Docker Desktop** instalado en tu sistema
  - Windows: [Descargar Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Asegúrate de que Docker esté ejecutándose antes de usar los comandos

## 🚀 Opciones de Ejecución

### Opción 1: Docker Compose (Recomendado)
La forma más sencilla de ejecutar la aplicación:

```bash
# Ejecutar el script de Docker Compose
docker-compose-run.bat
```

O manualmente:
```bash
docker compose up --build
```

### Opción 2: Docker Run Directo
```bash
# Ejecutar el script de Docker
docker-run.bat
```

O manualmente:
```bash
# Construir la imagen
docker build -t dashboard-meteorologico .

# Ejecutar el contenedor
docker run -p 8503:8503 -v "%cd%\datasets:/app/datasets:ro" -v "%cd%\datasets_limpios:/app/datasets_limpios:ro" dashboard-meteorologico
```

## 🌐 Acceso a la Aplicación

Una vez que el contenedor esté ejecutándose, abre tu navegador y ve a:
- **URL**: http://localhost:8503

## 📂 Estructura Docker

### Archivos Creados:
- **Dockerfile**: Configuración para construir la imagen Docker
- **docker-compose.yml**: Configuración para Docker Compose
- **.dockerignore**: Archivos excluidos del contexto de construcción
- **docker-run.bat**: Script para ejecutar con Docker directo
- **docker-compose-run.bat**: Script para ejecutar con Docker Compose

### Configuración del Contenedor:
- **Puerto**: 8503 (mapeado al puerto 8503 del host)
- **Volúmenes**: Los directorios `datasets` y `datasets_limpios` se montan como solo lectura
- **Reinicio**: El contenedor se reinicia automáticamente si falla
- **Health Check**: Verificación automática del estado del contenedor

## 🔧 Comandos Útiles

### Gestión de Contenedores:
```bash
# Ver contenedores en ejecución
docker ps

# Detener el contenedor
docker compose down

# Ver logs del contenedor
docker compose logs

# Ejecutar en segundo plano
docker compose up -d

# Reconstruir la imagen
docker compose build --no-cache
```

### Gestión de Imágenes:
```bash
# Listar imágenes
docker images

# Eliminar imagen
docker rmi dashboard-meteorologico

# Limpiar imágenes no utilizadas
docker image prune
```

## 🔍 Solución de Problemas

### Error: "Docker no está instalado"
- Instala Docker Desktop desde el enlace oficial
- Asegúrate de que Docker esté en el PATH del sistema

### Error: "Docker no está ejecutándose"
- Inicia Docker Desktop
- Espera a que Docker esté completamente iniciado (ícono en la bandeja del sistema)

### Error: "Puerto 8503 en uso"
- Detén otros contenedores: `docker compose down`
- O cambia el puerto en docker-compose.yml: `"8504:8503"`

### Error de permisos en volúmenes
- En Windows, asegúrate de que Docker tenga permisos para acceder a la carpeta del proyecto

## 🎯 Ventajas de la Versión Docker

1. **Portabilidad**: Ejecuta en cualquier sistema con Docker
2. **Consistencia**: Mismo entorno en desarrollo y producción
3. **Aislamiento**: No interfiere con otras aplicaciones
4. **Fácil despliegue**: Un solo comando para iniciar todo
5. **Escalabilidad**: Fácil de escalar horizontalmente
6. **Actualizaciones**: Solo rebuild la imagen para actualizaciones

## 📊 Monitoreo

El contenedor incluye un health check que verifica cada 30 segundos si Streamlit está respondiendo correctamente. Puedes ver el estado con:

```bash
docker ps
```

La columna STATUS mostrará "healthy" cuando todo funcione correctamente.

---

¡Tu Sistema de Predicciones Meteorológicas ahora está completamente dockerizado! 🐳🔮
