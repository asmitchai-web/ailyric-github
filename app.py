from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.generate_full_song import generate_song
import streamlit as st
import requests

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("src/model")
model = AutoModelForCausalLM.from_pretrained("src/model")

@app.post("/generate")
def generate():
    data = request.json
    lyrics = generate_song(
        model=model,
        tokenizer=tokenizer,
        genre=data["genre"],
        mood=data["mood"],
        structure=data["structure"],
        temperature=data["temperature"]
    )
    return jsonify({"lyrics": lyrics})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


st.title("AI Lyric Generator 🎵")

genre = st.selectbox("Genre", ["Pop", "Rock", "Hip-Hop"])
mood = st.selectbox("Mood", ["Happy", "Sad", "Energetic"])
structure = st.text_input("Structure", "V-C-V-C-B-C")
temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.9)

if st.button("Generate Lyrics"):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "genre": genre,
            "mood": mood,
            "structure": structure,
            "temperature": temperature
        }
    )
    st.text_area("Generated Lyrics", response.json()["lyrics"], height=400)

