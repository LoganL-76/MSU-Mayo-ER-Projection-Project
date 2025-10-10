import streamlit as st
import pandas as pd
import plotly.express as px
#st.title("Volume Prediction Model")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Training", "Predictions", "Page 3"])

with tab1:
    st.header("Training")
    st.write("This is the content of Training.")

with tab2:
    st.title("7-Day Patient Volume Forecast")

    # Example forecast list
    forecast = [
    ("10/9/2025", 22),
    ("10/10/2025", 25),
    ("10/11/2025", 20),
    ("10/12/2025", 28),
    ("10/13/2025", 30),
    ("10/14/2025", 26),
    ("10/15/2025", 24)
    ]

    # Convert to DataFrame
    df = pd.DataFrame(forecast, columns=["Date", "Patient Volume"])

    # Display as a table
    st.table(df)

    # Display as a line chart
    fig = px.line(df, x="Date", y="Patient Volume", markers=True, title="Patient Volume Forecast")
    st.plotly_chart(fig)

with tab3:
    st.header("Page 3")
    st.write("This is the content of Page 3.")
