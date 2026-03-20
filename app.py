import streamlit as st
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Health Early Warning System", layout="wide")

st.title("🚑 Smart Community Health Monitoring System")
st.subheader("Early Warning for Water-Borne Diseases")

# Sidebar Inputs
st.sidebar.header("📝 Health Worker Input")

diarrhea = st.sidebar.number_input("Diarrhea Cases", 0, 30, 5)
fever = st.sidebar.number_input("Fever Cases", 0, 30, 3)
vomiting = st.sidebar.number_input("Vomiting Cases", 0, 20, 2)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0, 300, 60)
temperature = st.sidebar.number_input("Temperature (°C)", 15, 45, 30)

if st.sidebar.button("Predict Risk"):
    input_data = [[diarrhea, fever, vomiting, rainfall, temperature]]
    risk = model.predict(input_data)[0]

    if risk == 0:
        st.success("🟢 LOW RISK – Situation Normal")
    elif risk == 1:
        st.warning("🟡 MEDIUM RISK – Monitor Closely")
    else:
        st.error("🔴 HIGH RISK – ALERT ISSUED")

        st.markdown("### 🚨 ALERT MESSAGE")
        st.code(
            "High risk of water-borne disease detected.\n"
            "Advise boiling water & immediate sanitation measures."
        )

# Trend Visualization
st.markdown("### 📊 Historical Trends")
df = pd.read_csv("health_data.csv")

st.line_chart(
    df[["diarrhea", "fever", "vomiting"]].tail(30)
)

st.markdown("### 🧠 AI Explanation")
st.info(
    "Risk prediction is based on sudden rise in symptoms combined "
    "with rainfall and temperature trends."
)
