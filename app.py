from flask import Flask, render_template, request, jsonify
import pyttsx3
import pickle
import sklearn
app = Flask(__name__)

# Load the trained ML model
with open("model (1).pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

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

    # Speak the response
    speak(response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
