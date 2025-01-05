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

st.set_page_config(
    page_title="Predicción de precios de acciones",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Predicciones usando un modelo híbrido")
st.markdown("""
Esta aplicación combina los modelos LSTM y regresión lineal para predicir precios (de cierre) de acciones.
Puede usarse subiendo datos históricos de acciones.
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

def make_future_predictions(lstm_model, lin_model, last_sequence, recent_data, scaler, days=10):
    lstm_future_predictions = []
    current_sequence = last_sequence.reshape(1, 30, 1)
    for _ in range(days):
        lstm_pred = lstm_model.predict(current_sequence, verbose=0)[0, 0]
        lstm_future_predictions.append(lstm_pred)
        lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
    
    
    lin_future_predictions = []
    current_data = recent_data.copy()
    for _ in range(days):
        lin_pred = lin_model.predict(current_data.reshape(1, -1))[0]
        lin_future_predictions.append(lin_pred)
        current_data = np.append(current_data[1:], lin_pred)
    

    lstm_future_predictions = scaler.inverse_transform(
        np.array(lstm_future_predictions).reshape(-1, 1)
    )
    lin_future_predictions = scaler.inverse_transform(
        np.array(lin_future_predictions).reshape(-1, 1)
    )
    
    
    hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)
    
    return lstm_future_predictions, lin_future_predictions, hybrid_future_predictions

def plot_predictions(data, future_dates, lstm_pred, lin_pred, hybrid_pred):
    fig = go.Figure()

    #Datos históricos
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Datos históricos',
        line=dict(color='gray')
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lstm_pred.flatten(),
        name='Predicciones con LSTM',
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lin_pred.flatten(),
        name='Predicciones con modelo lineal',
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=hybrid_pred.flatten(),
        name='Predicciones híbridas',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title='Predicciones de precios',
        xaxis_title='Fecha',
        yaxis_title='Precios',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data = data[['Close']]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        data['Close_Scaled'] = scaled_data

        seq_length = 30
        X, y = create_sequences(data['Close_Scaled'].values, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        data['Lag_1'] = data['Close_Scaled'].shift(1)
        data['Lag_2'] = data['Close_Scaled'].shift(2)
        data['Lag_3'] = data['Close_Scaled'].shift(3)
        data = data.dropna()

        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        X_train_lin = train_data[['Lag_1', 'Lag_2', 'Lag_3']][seq_length:]
        y_train_lin = train_data['Close_Scaled'][seq_length:]

        with st.spinner('Entrenando modelos...'):
            lstm_model = train_lstm_model(X_train, y_train)
            lin_model = train_linear_model(X_train_lin, y_train_lin)
            st.success('Se ha entrenado los modelos con éxito')

        
        st.subheader('Predecir')
        days = st.slider('Días por predecir:', 1, 30, 10)

        if st.button('Generar predicciones'):
            last_sequence = X[-1]
            recent_data = data['Close_Scaled'].values[-3:]

            lstm_pred, lin_pred, hybrid_pred = make_future_predictions(
                lstm_model, lin_model, last_sequence, recent_data, scaler, days
            )

            future_dates = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=days
            )

            fig = plot_predictions(data, future_dates, lstm_pred, lin_pred, hybrid_pred)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader('Predicciones')
            predictions_df = pd.DataFrame({
                'Fecha': future_dates,
                'Predicción LSTM': lstm_pred.flatten(),
                'Predicción modelo lineal': lin_pred.flatten(),
                'Predicción modelo híbrido': hybrid_pred.flatten()
            })
            predictions_df.set_index('Date', inplace=True)
            st.dataframe(predictions_df.round(2))

    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
else:
    st.info("Subir archivo CSV con los datos de las acciones")