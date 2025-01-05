import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Acciones",
    page_icon="📈",
    layout="wide"
)

# Título y descripción
st.title("📈 Predictor Híbrido de Precios de Acciones")
st.markdown("""
Esta aplicación combina modelos LSTM y Regresión Lineal para predecir precios de acciones.
¡Sube tus datos históricos de acciones para comenzar!
""")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, epochs=20):
    X_train = torch.FloatTensor(X_train).reshape(-1, 30, 1)
    y_train = torch.FloatTensor(y_train)
    
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
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
        st.subheader("Vista Previa de Datos")
        st.dataframe(data.head())
        
        # Mostrar columnas disponibles
        st.subheader("Columnas Disponibles")
        st.write(list(data.columns))
        
        # Selección de columnas
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
        st.write("Columnas originales:", data.columns.tolist())
        
        # Convertir fecha y establecer como índice
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        
        # Mantener solo el precio de cierre para el modelado
        if price_column in data.columns:
            model_data = data[[price_column]].copy()
        else:
            st.error(f"Columna {price_column} no encontrada en los datos")
            st.stop()
            
        st.write("Usando columna para predicción:", price_column)
        st.write("Forma de los datos:", model_data.shape)

        # Escalar datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(model_data)
        model_data[f'{price_column}_Scaled'] = scaled_data

        # Crear secuencias para LSTM
        seq_length = 30
        X, y = create_sequences(model_data[f'{price_column}_Scaled'].values, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Preparar datos para Regresión Lineal
        model_data[f'Lag_1'] = model_data[f'{price_column}_Scaled'].shift(1)
        model_data[f'Lag_2'] = model_data[f'{price_column}_Scaled'].shift(2)
        model_data[f'Lag_3'] = model_data[f'{price_column}_Scaled'].shift(3)
        model_data = model_data.dropna()

        train_size = int(len(model_data) * 0.8)
        train_data = model_data[:train_size]
        test_data = model_data[train_size:]

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
            # Predicciones LSTM
            lstm_model.eval()
            lstm_future_predictions = []
            last_sequence = torch.FloatTensor(X[-1]).reshape(1, 30, 1)
            
            for _ in range(days):
                with torch.no_grad():
                    pred = lstm_model(last_sequence).item()
                lstm_future_predictions.append(pred)
                last_sequence = torch.cat((
                    last_sequence[:, 1:, :],
                    torch.FloatTensor([[[pred]]])
                ), dim=1)
            
            # Predicciones lineales
            recent_data = model_data[f'{price_column}_Scaled'].values[-3:]
            lin_future_predictions = []
            current_data = recent_data.copy()
            
            for _ in range(days):
                pred = lin_model.predict(current_data.reshape(1, -1))[0]
                lin_future_predictions.append(pred)
                current_data = np.append(current_data[1:], pred)
            
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
                start=model_data.index[-1] + timedelta(days=1),
                periods=days
            )

            # Crear y mostrar gráfico
            fig = go.Figure()

            # Datos históricos
            fig.add_trace(go.Scatter(
                x=model_data.index,
                y=model_data[price_column],
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
        st.exception(e)
else:
    st.info("Por favor, sube un archivo CSV con datos de acciones para comenzar.")