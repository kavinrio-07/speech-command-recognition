import streamlit as st
import tensorflow as tf
import numpy as np
import librosa

# Load model
MODEL_PATH = "model/speech_commands_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

commands = ['down', 'go', 'left', 'no', 'off', 'on',
            'right', 'stop', 'up', 'yes', '_silence_', '_unknown_']

def get_spectrogram(waveform):
    waveform = tf.cast(waveform, tf.float32)
    waveform = tf.reshape(waveform, [-1])

    length = tf.shape(waveform)[0]

    waveform = tf.cond(
        length < 16000,
        lambda: tf.pad(waveform, [[0, 16000 - length]]),
        lambda: waveform[:16000]
    )

    spectrogram = tf.signal.stft(
        waveform,
        frame_length=255,
        frame_step=128
    )

    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)

    return spectrogram

def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    spec = get_spectrogram(audio)
    spec = tf.expand_dims(spec, 0)

    prediction = model.predict(spec)
    index = np.argmax(prediction)

    return commands[index], float(np.max(prediction))


# ---------------- STREAMLIT UI ----------------

st.title("🎙 Speech Command Recognition")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    label, confidence = predict_audio("temp.wav")

    st.success(f"Predicted Command: {label}")
    st.info(f"Confidence: {confidence:.2f}")
