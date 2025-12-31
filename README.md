# 🎙️ Speech Command Recognition using Deep Learning

This project is a Speech Command Recognition system built using Python, TensorFlow, and Streamlit.  
It allows users to upload a `.wav` audio file and predicts the spoken command with confidence.

---

## 🚀 Features
- Upload `.wav` audio files
- Converts audio to spectrogram
- Deep Learning model for prediction
- Displays predicted command and confidence
- Simple and clean Streamlit UI

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Streamlit

---

## 📂 Project Structure

speech_command_project/
│
├── app_streamlit.py                 # Streamlit web app  
├── requirements.txt                 # Python dependencies  
├── README.md                        # Project documentation  
│
├── model/  
│   └── speech_commands_model.keras  # Trained model  
│
├── sample_audio/  
│   └── yes_real.wav                 # Sample audio  
│
└── tnsdc.ipynb                      # Model training notebook  

---

## 🚀 Live Demo (Streamlit App)

🔗 https://speech-command-recognition.streamlit.app

Upload a `.wav` audio file and get the predicted speech command with confidence score.

---

## ▶️ How to Run the Project

## Activate Environment

conda activate speechcmds

## Install Dependencies

pip install -r requirements.txt

## Run the Streamlit App

streamlit run app_streamlit.py

## Open your browser and go to:
http://localhost:8501

---

## 🧪 How to Use the Application

- Open the Streamlit app in browser
- Upload a .wav audio file
- Model predicts the spoken command
- Confidence score is displayed

---

## 🧠 Model Details

- Dataset: TensorFlow Speech Commands
- Input: Audio converted to spectrogram
- Model: CNN based Deep Learning model
- Output: Command label with confidence

---

## 🚀 Future Improvements

- Real-time microphone input
- More commands support
- Cloud deployment
- Accuracy improvements

---

## 👤 Author
Kavin Raja
- Speech Command Recognition Project
