import streamlit as st
import whisper
import json
import numpy as np
import librosa
from transformers import pipeline
import requests
import zonos

model = whisper.load_model("tiny")

with open('dua.json', 'r', encoding='utf-8') as file:
    dua_data = json.load(file)

with open('verse.json', 'r', encoding='utf-8') as file:
    quranic_verses = json.load(file)

emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def zonos_tts(text, language="en"):
    url = "https://api.zonos.ai/tts"
    payload = {"text": text, "language": language}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("audio_url")
    return None

def analyze_emotion(text):
    result = emotion_analyzer(text)
    emotion_scores = {score['label']: score['score'] for score in result[0]}
    return max(emotion_scores, key=emotion_scores.get)

def get_duas(emotion):
    return dua_data.get(emotion, [])

def get_quranic_verses(emotion):
    return quranic_verses.get(emotion, [])

def load_audio_librosa(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return np.array(audio, dtype=np.float32)

st.title("Emotion-based Dua and Quranic Verse Suggestion")

st.markdown("### Upload your audio file (MP3 or WAV):")
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file:
    temp_file_path = "temp_audio_file.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_data = load_audio_librosa(temp_file_path)
    result = model.transcribe(audio_data)
    text = result["text"]

    st.markdown(f"#### Transcribed Text:")
    st.text_area("", text, height=150)

    emotion = analyze_emotion(text)
    st.markdown(f"<h3 style='color: #4CAF50;'>Detected Emotion: {emotion.capitalize()}</h3>", unsafe_allow_html=True)

    duas = get_duas(emotion)
    if duas:
        st.markdown("<h3 style='color: #FF7043;'>Suggested Duas:</h3>", unsafe_allow_html=True)
        for dua in duas:
            dua_audio_url = zonos_tts(dua['english'])
            st.markdown(
                f"""
                <div style='padding: 15px; border-radius: 8px; background-color: #2E3B4E; color: white; margin-bottom: 10px;'>
                <strong>Arabic:</strong> {dua['dua']}<br>
                <strong>Urdu:</strong> {dua['urdu']}<br>
                <strong>English:</strong> {dua['english']}</div>
                """, unsafe_allow_html=True
            )
            if dua_audio_url:
                st.audio(dua_audio_url, format='audio/mp3')
    else:
        st.write("No duas available for this emotion.")

    verses = get_quranic_verses(emotion)
    if verses:
        st.markdown("<h3 style='color: #FF7043;'>Suggested Quranic Verses:</h3>", unsafe_allow_html=True)
        for verse in verses:
            verse_audio_url = zonos_tts(verse['english'])
            st.markdown(
                f"""
                <div style='padding: 15px; border-radius: 8px; background-color: #2E3B4E; color: white; margin-bottom: 10px;'>
                <strong>Arabic:</strong> {verse['verse']}<br>
                <strong>Urdu:</strong> {verse['urdu']}<br>
                <strong>English:</strong> {verse['english']}</div>
                """, unsafe_allow_html=True
            )
            if verse_audio_url:
                st.audio(verse_audio_url, format='audio/mp3')
    else:
        st.write("No Quranic verses available for this emotion.")
