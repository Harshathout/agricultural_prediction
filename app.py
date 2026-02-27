
# 🌾 CropGuru AI – Groq Final Production Version


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import os

from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie



# LOAD ENVIRONMENT VARIABLES SAFELY

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USE_GROQ = True



# STREAMLIT CONFIG

st.set_page_config(
    page_title="CropGuru AI",
    page_icon="🌾",
    layout="wide"
)



# LANGUAGE TOGGLE

lang = st.radio(
    "🌐 Choose Language / భాషను ఎంచుకోండి",
    ["English", "తెలుగు"]
)



# TRANSLATIONS

t = {
    "title": {
        "English": "🌱 Smart Crops, Smart Farming - Profit Today, Success Tomorrow.",
        "తెలుగు": "🌱  స్మార్ట్ వ్యవసాయం - ఈరోజు లాభం, రేపటి విజయం."
    },
    "main_title": {"English": "🌾 CropGuru AI", "తెలుగు": "🌾 క్రాప్‌గురు AI"},
    "subtitle": {"English": "Smart Crop Prediction System", "తెలుగు": "స్మార్ట్ పంట ఊహించే వ్యవస్థ"},
    "recommend": {"English": "🌾 Crop Recommendation", "తెలుగు": "🌾 పంట సూచన"},
    "profit": {"English": "💸 Profit Calculator", "తెలుగు": "💸 లాభ లెక్కింపు"},
    "alt": {"English": "🌿 Alternative Crops", "తెలుగు": "🌿 ప్రత్యామ్నాయ పంటలు"},
    "chat": {"English": "🤖 Ask Agritech AI", "తెలుగు": "🤖 వ్యవసాయ AI ని అడగండి"},
    "chat_btn": {"English": "💬 Get AI Answer", "తెలుగు": "💬 AI సమాధానం"},
    "ai_resp": {"English": "🧠 AI Response", "తెలుగు": "🧠 AI సమాధానం"},
}



# TITLE

st.title(t["title"][lang])
st.header(t["main_title"][lang])
st.subheader(t["subtitle"][lang])


# LOAD DATASET

@st.cache_data
def load_data():
    return pd.read_csv("crop_data.csv")

data = load_data()

label_encoder = LabelEncoder()
data["CROP"] = label_encoder.fit_transform(data["CROP"])

X = data.drop(columns=["CROP"])
y = data["CROP"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



# GROQ API FUNCTION

def query_groq_model(question, lang):

    if not GROQ_API_KEY:
        return "__NO_API_KEY__"

    system_prompt = "You are an expert Agritech AI assistant helping Indian farmers."
    if lang == "తెలుగు":
        system_prompt += " Please respond clearly in Telugu."

    payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ],
    "temperature": 0.4 
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            return f"Groq Error {response.status_code}: {response.text}"

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return str(e)



# SAFE AI WRAPPER

def query_ai(question, lang):

    if not USE_GROQ:
        return "Demo AI response (Groq disabled)."

    result = query_groq_model(question, lang)

    if "__NO_API_KEY__" in str(result):
        return "⚠️ Groq API key not found. Please check your .env file."

    if "Groq Error" in str(result):
        return f"⚠️ AI service error:\n\n{result}"

    return result



# UI TABS

tab1, tab2, tab3, tab4 = st.tabs([
    t["recommend"][lang],
    t["profit"][lang],
    t["alt"][lang],
    t["chat"][lang]
])



# TAB 1 – Crop Recommendation

with tab1:
    col1, col2, col3 = st.columns(3)
    N = col1.number_input("Nitrogen (N)", min_value=0.0)
    P = col2.number_input("Phosphorus (P)", min_value=0.0)
    K = col3.number_input("Potassium (K)", min_value=0.0)

    col4, col5, col6 = st.columns(3)
    pH = col4.number_input("pH", min_value=0.0)
    rainfall = col5.number_input("Rainfall (mm)", min_value=0.0)
    temperature = col6.number_input("Temperature (°C)", min_value=0.0)

    if st.button("🔍 Predict Crops"):
        user_input = np.array([[N, P, K, pH, rainfall, temperature]])
        probs = rf_model.predict_proba(user_input)[0]
        top_idx = np.argsort(probs)[-3:][::-1]
        crops = label_encoder.inverse_transform(top_idx)

        st.success("🌾 Top Recommended Crops")
        for i, crop in enumerate(crops, 1):
            st.markdown(f"**{i}. {crop.title()}**")



# TAB 2 – Profit Calculator


# crop_data = {
#     "Rice": {"yield_per_acre": 2500, "price": 20, "cost": 30000},
#     "Wheat": {"yield_per_acre": 2000, "price": 22, "cost": 25000},
#     "Maize": {"yield_per_acre": 1800, "price": 18, "cost": 20000},
# }
# with tab2:
#     st.subheader("💰 Crop-Based Profit Calculator")

#     crop = st.selectbox("Select Crop", list(crop_data.keys()))
#     area = st.number_input("Area (acres)", min_value=0.0)

#     if st.button("🧮 Calculate Profit"):
#         if area == 0:
#             st.warning("Please enter a valid area.")
#         else:
#             yield_per_acre = crop_data[crop]["yield_per_acre"]
#             price = crop_data[crop]["price"]
#             cost_per_acre = crop_data[crop]["cost_per_acre"]

#             total_yield = area * yield_per_acre
#             total_cost = area * cost_per_acre
#             revenue = total_yield * price
#             profit = revenue - total_cost
#             per_acre_profit = profit / area

#             st.success(f"🌾 Total Yield: {total_yield:.2f} kg")
#             st.info(f"💵 Total Revenue: ₹{revenue:.2f}")
#             st.error(f"💸 Total Cost: ₹{total_cost:.2f}")
#             st.success(f"✅ Net Profit: ₹{profit:.2f}")
#             st.info(f"📊 Profit per Acre: ₹{per_acre_profit:.2f}")


with tab2:
    area = st.number_input("Area (acres)", min_value=0.0)
    yield_quintal = st.number_input("Expected Yield (quintal)", min_value=0.0)
    price = st.number_input("Market Price (₹/quintal)", min_value=0.0)
    cost = st.number_input("Total Cost (₹)", min_value=0.0)

    if st.button("🧮 Calculate"):
    
      if area == 0:
        st.warning("Please enter valid area.")
      else:
        revenue = (yield_quintal*100) * (price/100)
        profit = revenue - cost
        per_acre = profit / area

        st.success(f"Net Profit: ₹{profit:.2f}")
        st.info(f"Profit per Acre: ₹{per_acre:.2f}")




# TAB 3 – Alternative Crops

with tab3:
    crop_input = st.text_input("Enter Crop Name")

    if st.button("🔁 Find Alternatives"):
        crop_input = crop_input.strip().upper()
        crops = label_encoder.classes_

        if crop_input not in crops:
            st.error("Crop not found")
        else:
            label = label_encoder.transform([crop_input])[0]
            vec = data[data["CROP"] == label].drop(columns=["CROP"]).values[0]
            sim = cosine_similarity(X, vec.reshape(1, -1)).flatten()

            data["sim"] = sim
            alts = (
                data[data["CROP"] != label]
                .sort_values("sim", ascending=False)["CROP"]
                .unique()[:3]
            )

            alt_names = label_encoder.inverse_transform(alts)

            for alt in alt_names:
                st.markdown(f"### 🌱 {alt.title()}")
                st.write(query_ai(
                    f"Why is {alt} a good alternative to {crop_input}?",
                    lang
                ))



# TAB 4 – Chat with AI

with tab4:
    question = st.text_input("Type your farming question")

    if st.button(t["chat_btn"][lang]):
        if question.strip():
            st.subheader(t["ai_resp"][lang])
            st.write(query_ai(question, lang))
        else:
            st.warning("Please enter a question.")

