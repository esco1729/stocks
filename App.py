import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(
    page_title="Predictor de Precios de Acciones",
    page_icon="📈",
    layout="wide"
)

# Título y descripción
st.title("📈 Predictor Híbrido de Precios de Acciones")
st.markdown("""
Esta aplicación combina modelos LSTM y Regresión Lineal para predecir precios de acciones.
""")

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model

def train_linear_model(X_train_lin, y_train_lin):
    model = LinearRegression()
    model.fit(X_train_lin, y_train_lin)
    return model

# Cargador de archivos
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Cargar datos
        data = pd.read_csv(uploaded_file)
        
        # Mostrar vista previa de datos
        st.subheader("Vista previa de datos sin procesar")
        st.dataframe(data.head())
        
        st.subheader("Columnas Disponibles")
        st.write(list(data.columns))
        
        date_column = st.selectbox(
            "Selecciona la columna de fecha",
            options=data.columns,
            index=0 if 'Date' in data.columns else 0
        )
        
        price_column = st.selectbox(
            "Selecciona la columna de precio",
            options=data.columns,
            index=data.columns.get_loc('Close') if 'Close' in data.columns else 0
        )
        
        # Procesar datos
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        data = data[[price_column]]
        
        # Escalar datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        data[f'{price_column}_Scaled'] = scaled_data

        # Crear secuencias para LSTM
        seq_length = 30
        X, y = create_sequences(data[f'{price_column}_Scaled'].values, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Preparar datos para Regresión Lineal
        data[f'Lag_1'] = data[f'{price_column}_Scaled'].shift(1)
        data[f'Lag_2'] = data[f'{price_column}_Scaled'].shift(2)
        data[f'Lag_3'] = data[f'{price_column}_Scaled'].shift(3)
        data = data.dropna()

        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        X_train_lin = train_data[[f'Lag_1', f'Lag_2', f'Lag_3']][seq_length:]
        y_train_lin = train_data[f'{price_column}_Scaled'][seq_length:]

        # Entrenar modelos
        with st.spinner('Entrenando modelos...'):
            lstm_model = train_lstm_model(X_train, y_train)
            lin_model = train_linear_model(X_train_lin, y_train_lin)
            st.success('¡Modelos entrenados exitosamente!')

        # Sección de predicciones
        st.subheader('Realizar Predicciones')
        days = st.slider('Selecciona el número de días a predecir:', 1, 30, 10)

        if st.button('Generar Predicciones'):
            # Obtener última secuencia para LSTM
            last_sequence = X[-1]
            # Obtener datos recientes para Regresión Lineal
            recent_data = data[f'{price_column}_Scaled'].values[-3:]

            # Predicciones LSTM
            lstm_future_predictions = []
            current_sequence = last_sequence.reshape(1, 30, 1)
            for _ in range(days):
                lstm_pred = lstm_model.predict(current_sequence, verbose=0)[0, 0]
                lstm_future_predictions.append(lstm_pred)
                lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
                current_sequence = np.append(current_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
            
            # Predicciones lineales
            lin_future_predictions = []
            current_data = recent_data.copy()
            for _ in range(days):
                lin_pred = lin_model.predict(current_data.reshape(1, -1))[0]
                lin_future_predictions.append(lin_pred)
                current_data = np.append(current_data[1:], lin_pred)
            
            # Convertir predicciones a escala original
            lstm_future_predictions = scaler.inverse_transform(
                np.array(lstm_future_predictions).reshape(-1, 1)
            )
            lin_future_predictions = scaler.inverse_transform(
                np.array(lin_future_predictions).reshape(-1, 1)
            )
            
            # Predicciones híbridas
            hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)

            # Generar fechas futuras
            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=days
            )

            # Crear y mostrar gráfico
            fig = go.Figure()

            # Datos históricos
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[price_column],
                name='Datos Históricos',
                line=dict(color='gray')
            ))

            # Predicciones
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lstm_future_predictions.flatten(),
                name='Predicciones LSTM',
                line=dict(dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lin_future_predictions.flatten(),
                name='Predicciones Lineales',
                line=dict(dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=hybrid_future_predictions.flatten(),
                name='Predicciones Híbridas',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title='Predicciones de Precios de Acciones',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                hovermode='x unified',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Mostrar tabla de predicciones
            st.subheader('Valores Predichos')
            predictions_df = pd.DataFrame({
                'Fecha': future_dates,
                'Predicción LSTM': lstm_future_predictions.flatten(),
                'Predicción Regresión Lineal': lin_future_predictions.flatten(),
                'Predicción Modelo Híbrido': hybrid_future_predictions.flatten()
            })
            predictions_df.set_index('Fecha', inplace=True)
            st.dataframe(predictions_df.round(2))

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("Por favor, sube un archivo CSV con datos de acciones para comenzar.")