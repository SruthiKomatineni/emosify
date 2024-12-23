import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background: rgb(233,113,165);
            background: linear-gradient(300deg, rgba(233,113,165,1) 0%, rgba(173,204,241,1) 100%);
            background-size: cover;
        }
        .center-image {
            display: flex;
            justify-content: center;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)



#st.image("C:/Users/Hp/Downloads/allemoji_nobgm.png")
#st.image("C:/Users/Hp/Downloads/allemojis2_nobgm.png")


st.markdown('<div class="center-image">', unsafe_allow_html=True)
st.image("C:/Users/Hp/Downloads/allemoji_nobgm.png", width=750)  # Adjust width as needed
st.markdown('</div>', unsafe_allow_html=True)



# Load the pre-trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Application title
st.header("Emosify Music")

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Try to load previous emotion if available
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Emotion Processor class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        # Extract landmarks for emotion prediction
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion.npy", np.array([pred]))

        # Draw landmarks
        drawing.draw_landmarks(
            frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1)
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Language and Singer input
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Start the video stream if necessary
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Recommend songs button
btn = st.button("Recommend me songs")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+songs+by+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"


