import streamlit as st
import cv2
import mediapipe as mp
import google.generativeai as genai

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def analyze_video(video_path):
    """Analyze pickleball video using MediaPipe"""
    cap = cv2.VideoCapture(video_path)
    poses = []
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                poses.append(results.pose_landmarks)
    cap.release()
    return poses

def get_coaching_feedback(pose_data):
    """Generate feedback using Gemini"""
    genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    Analyze this pickleball player's pose data: {str(pose_data[:3])}...
    Provide 3 specific coaching tips focusing on:
    - Stance and balance
    - Swing mechanics
    - Ready position
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("🏓 Pickleball AI Coach")
uploaded_file = st.file_uploader("Upload a short pickleball video (5-10 seconds)", type=["mp4"])

if uploaded_file:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(video_path)
    
    if st.button("Analyze My Form"):
        with st.spinner("Analyzing your technique..."):
            poses = analyze_video(video_path)
            feedback = get_coaching_feedback(poses)
            st.subheader("Your Personalized Feedback")
            st.write(feedback)