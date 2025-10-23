import streamlit as st

from predict import make_prediction

st.title("Iris prediction app")

f1 = st.slider(label='SepalLengthCm',min_value=4.3,max_value=7.9)
f2 = st.slider(label='SepalWidthCm',min_value=2.0,max_value=4.4)
f3 = st.slider(label='PetalLengthCm',min_value=1.0,max_value=6.9)
f4 = st.slider(label='PetalWidthCm',min_value=1.0,max_value=2.5)

prediction = make_prediction([[f1, f2, f3, f4]])
st.markdown(f'### Prediction: {prediction}')