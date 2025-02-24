from flask import Flask, render_template, request, jsonify
import pickle
import sklearn
from gtts import gTTS  # Replacing pyttsx3 (Linux-compatible)
import os

app = Flask(__name__)

# Load the trained ML model
with open("model.pkl", "rb") as f:  # Fixed file name (No spaces)
    model = pickle.load(f)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def speak(text):
    """Convert text to speech using gTTS (Google TTS)"""
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")  # Works on Linux

@app.route("/")
def index():
    """Render the chatbot UI"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle user messages"""
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"response": "Please provide a message."})

    # Convert text input using the loaded vectorizer
    user_input_tfidf = vectorizer.transform([user_input])
    
    # Predict response using ML model
    response = model.predict(user_input_tfidf)[0]

    # Speak the response (Text-to-Speech)
    speak(response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
