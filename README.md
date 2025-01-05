# Predicciones de precios de acciones

Modelo híbrido de aprendizaje automático que combina LSTM y Regresión Lineal para predecir precios de acciones. La aplicación está construida con Streamlit y proporciona una interfaz interactiva para cargar datos de acciones y generar predicciones.

## Características
- Modelo de predicción híbrido que combina LSTM y Regresión Lineal
- Visualización interactiva de datos
- Marco temporal de predicción personalizable
- Interfaz sencilla para cargar archivos
- Resultados detallados de predicción


## Instalación
```bash
# Clonar el repositorio
git clone https://github.com/esco1729/stocks.git

# Navegar al directorio del proyecto
cd stocks

# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso
1. Ejecutar la aplicación Streamlit:
```bash
streamlit run app.py
```
2. Cargar un archivo CSV con datos de acciones (columnas requeridas: 'Date', 'Close')
3. Ajustar el marco temporal de predicción usando el control deslizante
4. Hacer clic en 'Generar Predicciones' para ver los resultados

## Tecnologías utilizadas
- Python
- Streamlit
- TensorFlow
- scikit-learn
- Plotly
- Pandas
- NumPy
