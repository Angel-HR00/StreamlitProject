from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import ta

#Settings
max_width = 1000
ta_col_prefix = 'ta_'


#Side-bar
return_value = st.sidebar.selectbox(
    "Numero de periodos para calcular el precio",
    [1, 2, 3, 5, 7, 14, 31]
)

#Preparar data
@st.cache(ignore_hash = True)
def load_data():
    
    #Cargar dataset
    df = pd.read_csv("data/processed/data.csv")
    
    #Limpiar valores nulos
    df. ta.utils.dropna(df)
    
    df = ta.add_volatility_ta(df, "High", "Low", "Close", fillna=False, colprefix=ta_col_prefix)
    df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume_Currency", fillna=False, colprefix=ta_col_prefix)
    
    return df

df = load_data()

df['y'] = (df['Close'] / df['Close'].shift(return_value) - 1) * 100

#Limpiar valores nulos
df =df.dropna(df)

#Streamlit
st.title("EDA para datos financieros")

a = datetime.utcfromtimestamp(df['Timestamp'].head(1)).strftime('%Y-%m-%d %H:%M:%S')
b = datetime.utcfromtimestamp(df['Timestamp'].tail(1)).strftime('%Y-%m-%d %H:%M:%S')

st.write(f"Intentamos explorar un pequeño lapso de tiempo con precios BTC/USD desde {a} hasta {b}")
st.write('')

st.subheader('Dataframe')
st.write(df)

st.subheader('Describe dataframe')
st.write(df.describe())

st.write('Number of rows: {}, Number of columns: {}'.format(*df.shape))

st.subheader('Price')
st.line_chart(df['Close'], width=max_width)

st.subheader(f'Return {return_value} periods')
st.area_chart(df['y'], width=max_width)

st.subheader('Target Histogram')
bins = list(np.arange(-10, 10, 0.5))
hist_values, hist_indexes = np.histogram(df['y'], bins=bins)
st.bar_chart(pd.DataFrame(data=hist_values, index=hist_indexes[0:-1]), width=max_width)
st.write('Target value min: {0:.2f}%; max: {1:.2f}%; mean: {2:.2f}%; std: {3:.2f}'.format(
    np.min(df['y']), np.max(df['y']), np.mean(df['y']), np.std(df['y'])))

# Analisis univariante
st.subheader('Correlation coefficient ta features and target column')

x_cols = [col for col in df.columns if col not in ['Timestamp', 'y'] and col.startswith(ta_col_prefix)]
labels = [col for col in x_cols]
values = [np.corrcoef(df[col], df['y'])[0, 1] for col in x_cols]

st.bar_chart(data=pd.DataFrame(data=values, index=labels), width=max_width)