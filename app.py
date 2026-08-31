import os
import tempfile
import cv2
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import whisper
import base64
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal Depression Detection",
    page_icon="🧠",
    layout="wide",
)

# --- Generate Custom Video Component ---
COMP_DIR = "custom_video_component"
os.makedirs(COMP_DIR, exist_ok=True)
html_path = os.path.join(COMP_DIR, "index.html")

html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; background: #1e1e1e; padding: 10px; color: #fff; margin: 0; }
        video { width: 480px; height: 320px; border-radius: 8px; background: #000; margin-bottom: 15px; border: 2px solid #444; }
        .btn-group { display: flex; gap: 10px; margin-bottom: 15px; }
        button { padding: 10px 20px; font-weight: bold; color: white; border: none; border-radius: 6px; cursor: pointer; }
        #startBtn { background-color: #ff4b4b; }
        #startBtn:disabled { background-color: #666; cursor: not-allowed; }
        #stopBtn { background-color: #666; cursor: not-allowed; }
        #stopBtn:enabled { background-color: #222; border: 1px solid #fff; cursor: pointer; }
        #status { font-size: 14px; color: #aaa; font-weight: bold; }
    </style>
</head>
<body>
    <video id="vid" autoplay playsinline muted></video>
    <div class="btn-group">
        <button id="startBtn">🔴 Start Recording</button>
        <button id="stopBtn" disabled>⏹️ Stop & Upload</button>
    </div>
    <div id="status">Ready to record</div>

    <script>
        function sendMessage(type, data) {
            window.parent.postMessage({ isStreamlitMessage: true, type: type, ...data }, "*");
        }
        sendMessage("streamlit:componentReady", {apiVersion: 1});
        sendMessage("streamlit:setFrameHeight", {height: 450});

        let mediaRecorder;
        let recordedChunks = [];
        const vid = document.getElementById('vid');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        navigator.mediaDevices.getUserMedia({ video: true, audio: true })
            .then(stream => { vid.srcObject = stream; window.localStream = stream; })
            .catch(err => { status.innerText = "Error: " + err; status.style.color = "#ff4b4b"; });

        startBtn.onclick = () => {
            recordedChunks = [];
            mediaRecorder = new MediaRecorder(window.localStream, {mimeType: 'video/webm'});
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
            
            mediaRecorder.onstop = () => {
                status.innerText = "Uploading to Python Backend... Please wait.";
                status.style.color = "#0088cc";
                
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onloadend = () => {
                    sendMessage("streamlit:setComponentValue", {value: reader.result});
                    status.innerText = "✅ Upload Complete! Scroll down to analyze.";
                    status.style.color = "#00ff88";
                };
            };
            
            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.innerText = "Recording...";
            status.style.color = "#ff4b4b";
        };

        stopBtn.onclick = () => {
            mediaRecorder.stop();
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };
    </script>
</body>
</html>
"""
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

live_video_recorder = components.declare_component("live_video_recorder", path=COMP_DIR)


# --- Load Models ---
@st.cache_resource
def load_models():
    text_model = joblib.load("depression_detection_models/text_depression_model.pkl")
    image_model = tf.keras.models.load_model("depression_detection_models/facial_emotion_model.keras")
    emotion_labels = joblib.load("depression_detection_models/emotion_labels.pkl")
    label_map = {v: k for k, v in emotion_labels.items()}
    whisper_model = whisper.load_model("base")
    return text_model, image_model, label_map, whisper_model


st.title("🧠 Multimodal Depression Detection")
st.write("Analyze text sentiment, voice recordings, facial expressions, and video recordings for depressive markers.")

with st.spinner("Loading AI models into memory..."):
    text_model, image_model, label_map, whisper_model = load_models()

# --- Helper: Calculate 1-10 Scale ---
def get_severity_level(probability):
    # Maps a 0.0 - 1.0 probability float to a 1 - 10 integer scale
    return int(probability * 9) + 1

def display_severity_ui(level):
    st.write(f"### 📊 Estimated Depression Severity: **{level}/10**")
    # Color progress bar based on severity
    if level <= 3:
        st.progress(level / 10.0)
        st.success("✅ Risk level is low. No significant depressive markers detected.")
    elif level <= 7:
        st.progress(level / 10.0)
        st.warning("⚠️ Risk level is moderate. Some emotional distress detected.")
    else:
        st.progress(level / 10.0)
        st.error("🚨 Risk level is high. Significant depressive markers detected.")

# --- Helper: Transcribe & Classify Speech ---
def process_media_audio(file_bytes, file_suffix):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("Transcribing speech with Whisper..."):
            result = whisper_model.transcribe(tmp_path, fp16=False)

        transcribed_text = result.get("text", "").strip()

        if transcribed_text:
            # predict_proba returns [[prob_0, prob_1]] - we want the probability of class 1 (depression)
            prob = text_model.predict_proba([transcribed_text])[0][1]
            severity = get_severity_level(prob)
            
            st.write("---")
            st.write("### 📝 Transcribed Text:")
            st.info(f'"{transcribed_text}"')
            
            display_severity_ui(severity)
        else:
            st.warning("No clear speech could be detected in the provided media.")

    except Exception as e:
        st.error(f"Error processing media: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --- UI Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Text Analysis",
    "🎤 Voice Record",
    "📸 Face / Image Analysis",
    "🎥 Video (Record & Upload)"
])

# 1. Text Analysis
with tab1:
    st.subheader("Text Sentiment & Semantic Analysis")
    user_text = st.text_area("Enter or paste text statements below:", height=150)
    if st.button("Analyze Text", key="btn_text"):
        if user_text.strip():
            prob = text_model.predict_proba([user_text])[0][1]
            severity = get_severity_level(prob)
            display_severity_ui(severity)
        else:
            st.warning("Please enter text before analyzing.")

# 2. Voice Record
with tab2:
    st.subheader("Live Voice Recording")
    audio_value = st.audio_input("Record Voice")
    if audio_value is not None:
        st.audio(audio_value)
        if st.button("Analyze Voice Recording", key="btn_voice"):
            process_media_audio(audio_value.getvalue(), ".wav")

# 3. Image Upload & Live Camera Capture
with tab3:
    st.subheader("Facial Emotion & Expression Analysis")
    img_source = st.radio("Choose image input method:", ["📸 Take Photo (Webcam)", "📂 Upload Image File"], horizontal=True)

    uploaded_img = None
    if img_source == "📸 Take Photo (Webcam)":
        uploaded_img = st.camera_input("Take a clear photo of your face")
    else:
        uploaded_img = st.file_uploader("Upload a clear image of a face", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(uploaded_img, caption="Target Image", width=250)

        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img_resized = cv2.resize(img, (48, 48))
            img_norm = img_resized.reshape(1, 48, 48, 1) / 255.0
            preds = image_model.predict(img_norm)[0]
            
            # Calculate severity based on sum of distress emotion probabilities
            distress_prob = sum(preds[k] for k, v in label_map.items() if v in ['sad', 'fear', 'angry'])
            severity = get_severity_level(distress_prob)
            
            top_class_idx = int(np.argmax(preds))
            detected_emotion = label_map[top_class_idx]
            
            with col_res:
                st.write(f"### Detected Primary Emotion: **{detected_emotion.capitalize()}**")
                display_severity_ui(severity)

# 4. Video (Live Recorder & Upload)
with tab4:
    st.subheader("Live Video Recording & Analysis")
    video_mode = st.radio("Choose Video Mode:", ["🔴 Live Camera Video Recorder", "📂 Upload Video File"], horizontal=True)

    if video_mode == "🔴 Live Camera Video Recorder":
        st.write("### Record Live Video & Audio")
        video_data_b64 = live_video_recorder(key="vid_recorder")

        if video_data_b64 is not None and isinstance(video_data_b64, str):
            if "," in video_data_b64:
                base64_str = video_data_b64.split(",", 1)[1]
                video_bytes = base64.b64decode(base64_str)
                
                st.write("---")
                st.write("### 📼 Your Captured Video")
                st.video(video_bytes)
                
                if st.button("Extract Audio & Analyze Speech", key="btn_live_vid_analyze"):
                    process_media_audio(video_bytes, ".webm")

    else:
        st.write("### 📂 Upload Recorded Video")
        uploaded_video = st.file_uploader("Upload an existing video", type=["mp4", "mov", "avi", "webm"])

        if uploaded_video is not None:
            st.video(uploaded_video)
            if st.button("Extract Audio & Analyze Speech", key="btn_upload_vid"):
                ext = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                process_media_audio(uploaded_video.getvalue(), ext)