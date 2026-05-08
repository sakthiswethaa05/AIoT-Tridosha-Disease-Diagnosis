import streamlit as st
import requests

from tongue.predict import predict
from voice.voice import get_voice_output

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.title("AIoT-Based Health Monitoring System")

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns(3)

name = col1.text_input("Patient Name")
age = col2.number_input("Age", min_value=1)
gender = col3.selectbox("Gender", ["Male", "Female"])

st.divider()

col4, col5 = st.columns(2)

tongue_img = col4.file_uploader("Upload Tongue Image", type=["jpg", "png"])
voice_file = col5.file_uploader("Upload Voice", type=["wav", "mp3"])

# ---------------- PULSE + VOICE DATA ----------------
pulse_info = {
    "Vata": "Headache, hypertension, anxiety, dry cough, sore throat, earache, insomnia, abdominal gas, diarrhea, abnormal heart rhythm, constipation, muscular spasms, lower back pain, arthritis, nervous system disorders",
    "Pitta": "Heartburn, ulcers, burning sensation in stomach or intestine, bowel disorders, skin rashes, inflammation, skin diseases, anemia, gallbladder disorders, liver diseases",
    "Kapha": "Obesity, allergies, cold, respiratory problems, congestion, sinus pain, asthma, atherosclerosis"
}

voice_info = {
    "Vata": "Dry voice, instability",
    "Pitta": "Sharp tone, irritation",
    "Kapha": "Heavy voice"
}

# ---------------- TONGUE SYMPTOMS ----------------
tongue_symptoms = {
    "PALE": {
        "Vata": "Fatigue, weakness, cold intolerance, dizziness",
        "Pitta": "-",
        "Kapha": "Sluggish metabolism, lethargy, low appetite"
    },
    "RED": {
        "Vata": "-",
        "Pitta": "Burning sensation, irritability, loose stools, headache",
        "Kapha": "-"
    },
    "PURPLE": {
        "Vata": "Poor circulation, numbness, stagnation",
        "Pitta": "Inflammatory congestion, blood heat",
        "Kapha": "Fluid retention, heaviness"
    },
    "WHITE_COAT": {
        "Vata": "-",
        "Pitta": "-",
        "Kapha": "Bloating, mucus formation, heaviness, lethargy"
    },
    "YELLOW_COAT": {
        "Vata": "-",
        "Pitta": "Hyperacidity, burning chest, skin rashes, anger",
        "Kapha": "-"
    },
    "BLACK_COAT": {
        "Vata": "Severe dryness, hard stools, irregular bowel",
        "Pitta": "Toxic heat, foul breath",
        "Kapha": "-"
    },
    "SPOTS": {
        "Vata": "Dark spots → poor circulation, anxiety",
        "Pitta": "Red spots → headache, inflammation, heavy menstruation",
        "Kapha": "-"
    },
    "CRACKED": {
        "Vata": "Constipation, insomnia, nervous tension, dryness",
        "Pitta": "Dehydration with heat (if red base)",
        "Kapha": "-"
    }
}

# ---------------- BUTTON ----------------
if st.button("Analyze"):

    if tongue_img and voice_file:

        # Save files
        with open("t.jpg", "wb") as f:
            f.write(tongue_img.read())

        with open("v.wav", "wb") as f:
            f.write(voice_file.read())

        # ---------------- AI OUTPUT ----------------
        tongue_result = predict("t.jpg")
        tongue = tongue_result["dosha"]
        tclass = tongue_result["class"]

        voice = get_voice_output("v.wav")["dosha"]

        # ---------------- PULSE ----------------
        try:
            with open("pulse_output.txt", "r") as f:
                pulse = f.read().strip()
        except:
            pulse = "Not Available"

        st.divider()

        # ---------------- CLINICAL DETAILS ----------------
        st.header("Clinical Details")

        colA, colB = st.columns(2)

        # -------- Tongue --------
        with colA:
            st.subheader("Tongue Findings")
            st.write("Class:", tclass)

            if tclass in tongue_symptoms:
                st.markdown("**Vata:** " + tongue_symptoms[tclass]["Vata"])
                st.markdown("**Pitta:** " + tongue_symptoms[tclass]["Pitta"])
                st.markdown("**Kapha:** " + tongue_symptoms[tclass]["Kapha"])
            else:
                st.write("No data available")

        # -------- Pulse + Voice --------
        with colB:
            st.subheader("Physiological Indicators")

            st.write("Pulse:", pulse)

            if pulse == "Not Available":
                st.warning("Pulse device not connected")
            else:
                st.write(pulse_info.get(pulse, "No data"))

            st.write("Voice:", voice)
            st.write(voice_info.get(voice, "No data"))

        # ---------------- PATIENT SUMMARY ----------------
        st.divider()
        st.header("Patient Summary")

        colX, colY = st.columns(2)

        with colX:
            st.write("Name:", name)
            st.write("Age:", age)
            st.write("Gender:", gender)

        with colY:
            st.write("Tongue:", tongue)
            st.write("Pulse:", pulse)
            st.write("Voice:", voice)

        # ---------------- CLOUD ----------------
        api_key = "TP511DFYBPB0TBTC"

        url = f"https://api.thingspeak.com/update?api_key={api_key}&field1={tongue}&field2={pulse}&field3={voice}"
        requests.get(url)

        st.success("Data stored successfully")

    else:
        st.error("Please upload both files")