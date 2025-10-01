import streamlit as st
import plotly.express as px

st.title("ED Patient Predictor")

# Input form
st.sidebar.header("Input Features")
date = st.sidebar.date_input("Date")
st.sidebar.number_input("Temperature (°F)")
st.sidebar.number_input("Precipitation (inches)")

# Replace with model output later
data = {
    'Diagnosis': ['Heart', 'Mental', 'Cancer'],
    'Percent': [45, 30, 25] 
}
fig = px.pie(data, names='Diagnosis', values='Percent', title='Diagnoses')

# Dummy prediction
if st.sidebar.button("Predict"):
    st.subheader("Predicted Patient Volume")
    st.metric("Patients", "15")  # Replace with model output later
    st.subheader("Predicted Patient Diagnoses")
    st.plotly_chart(fig)

