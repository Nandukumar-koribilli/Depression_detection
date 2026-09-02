# 🧠 Multimodal Depression Detection

A comprehensive Streamlit web application that analyzes text, voice, facial expressions, and video recordings to detect semantic and physiological markers of depression. The system uses a multimodal approach combining Natural Language Processing (NLP), Computer Vision (CV), and Audio processing.

## 🌟 Features

* **💬 Text Analysis:** TF-IDF & Logistic Regression model to detect depressive semantic markers.
* **🎤 Live Voice Recording:** OpenAI Whisper for speech-to-text extraction, followed by NLP sentiment analysis.
* **📸 Facial Emotion Recognition:** Real-time webcam capture or image upload processed by a custom Convolutional Neural Network (CNN) to detect distress (sadness, fear, anger).
* **🎥 Video Processing:** Custom bidirectional HTML5/JS component for live video recording, or file upload, to extract audio tracks and analyze spoken dialogue.
* **📊 Severity Scale:** 1-10 severity metric based on combined model probability outputs.

---

## ⚙️ Prerequisites

1. **Python 3.8 - 3.11** installed on your system.
2. **Pre-trained Models:** You must have the `depression_detection_models` folder (trained via Google Colab) containing the `.pkl` and `.keras` model artifacts.

---

## 🛠️ Installation

### 1. Set up the Directory

Create a main project folder and ensure your files are structured exactly like this:

```text
Depression detection/
│
├── app.py
├── README.md
├── .streamlit/
│   └── config.toml
└── depression_detection_models/
    ├── text_depression_model.pkl
    ├── facial_emotion_model.keras
    └── emotion_labels.pkl
```

### 2. Install Python Dependencies

Open your terminal (PowerShell, Command Prompt, or VS Code Terminal) inside your project folder and run:

```bash
python -m pip install --upgrade pip
python -m pip install streamlit opencv-python numpy scikit-learn tensorflow openai-whisper
```

> [!NOTE]
> Streamlit version must be **1.40.0 or higher** to support the `st.audio_input` widget.

### 3. Install FFmpeg (Required for Audio/Video)

OpenAI Whisper requires FFmpeg to extract audio tracks from media files.

**For Windows:**

Open PowerShell and run:

```powershell
winget install Gyan.FFmpeg
```

After installation, completely close and reopen your terminal so the system recognizes the new path.

**For macOS:**

```bash
brew install ffmpeg
```

**For Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install ffmpeg
```

---

## 🚀 Running the Application

1. Open a terminal in your project directory.

2. Launch the Streamlit server:

```bash
python -m streamlit run app.py
```

3. The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🔧 Troubleshooting

### `AttributeError: module 'streamlit' has no attribute 'audio_input'`

Your Streamlit version is outdated. Run:

```bash
python -m pip install --upgrade streamlit
```

---

### `AxiosError: Request failed with status code 403` on file upload

Create a `.streamlit/config.toml` file in your project root with the following to bypass CORS/XSRF blocks:

```toml
[server]
