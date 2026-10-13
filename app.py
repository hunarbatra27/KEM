
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import uuid
import speech_recognition as sr
import os

app = Flask(__name__)

# Allow your React website to communicate with Flask.
# For development, allow all origins.
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# LOAD MODEL
# ============================================================

vectorizer = joblib.load("tfidf_vectorizer.pkl")

qna_df = pd.read_csv("qna_data.csv")

question_vectors = vectorizer.transform(
    qna_df["question"]
)


# ============================================================
# CHATBOT ANSWER GENERATOR
# ============================================================

def get_answer(user_question):

    user_vec = vectorizer.transform([user_question])

    similarity_scores = cosine_similarity(
        user_vec,
        question_vectors
    )

    best_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_index]

    if best_score < 0.20:
        return "Sorry, I could not find a suitable answer."

    return str(qna_df.iloc[best_index]["answer"])


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return jsonify({
        "status": "success",
        "message": "API is working!"
    })


# ============================================================
# WEB CHATBOT
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Invalid or missing JSON"
        }), 400

    user_question = str(
        data.get("question", "")
    ).strip()

    if not user_question:
        return jsonify({
            "error": "No question provided"
        }), 400

    answer = get_answer(user_question)

    return jsonify({
        "answer": answer
    })


# ============================================================
# TEXT TO SPEECH
# ============================================================

@app.route("/robot_speak", methods=["POST"])
def robot_speak():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Invalid or missing JSON"
        }), 400

    user_question = str(
        data.get("question", "")
    ).strip()

    if not user_question:
        return jsonify({
            "error": "No question provided"
        }), 400

    answer = get_answer(user_question)

    filename = f"robot_{uuid.uuid4().hex}.mp3"

    try:
        tts = gTTS(text=answer, lang="en")
        tts.save(filename)

        return send_file(
            filename,
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ROBOT AUDIO QUESTION
# ============================================================

@app.route("/robot_ask", methods=["POST"])
def robot_ask():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file uploaded"
        }), 400

    audio_file = request.files["audio"]

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        user_question = recognizer.recognize_google(
            audio_data
        )

        print("User asked:", user_question)

        answer = get_answer(user_question)

        print("Answer:", answer)

        filename = f"robot_answer_{uuid.uuid4().hex}.mp3"

        tts = gTTS(text=answer, lang="en")
        tts.save(filename)

        return send_file(
            filename,
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except sr.UnknownValueError:
        return jsonify({
            "error": "Could not understand audio"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# SPEECH TO TEXT
# ============================================================

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file uploaded"
        }), 400

    audio_file = request.files["audio"]

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return jsonify({
            "text": text
        })

    except sr.UnknownValueError:
        return jsonify({
            "error": "Could not understand audio"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )