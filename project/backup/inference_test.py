import logging
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
from keras.saving.save import load_model
from streamlit_webrtc import webrtc_streamer

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
model = load_model("model.h5")


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    # print(results)
    # todo: add more classes of actions
    if results[0][0] > 0.5:
        label = "Hello"
    else:
        label = "Thank"
    return label


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


# ===================

lock = threading.Lock()
img_container = {"img": None}

# logging
st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.WARNING)


label = "Running ...."

# Number of frames to guess
n_time_steps = 10

# landmarks list. will be reset every guess
lm_list = []

def video_frame_callback(frame):
    global lm_list # global to access outside of method ???

    # copy from streamlit
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    # convert to RGB as Mediapipe's format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    c_lm = make_landmark_timestep(results)
    lm_list.append(c_lm)

    # when enough frame
    if len(lm_list) == n_time_steps:

        # new thread will handle the detection
        t1 = threading.Thread(target=detect, args=(model, lm_list,))
        t1.start()
        lm_list = []

    # draw landmark on frame
    img = draw_landmark_on_image(mpDraw, results, img)

    # draw prediction on image
    img = draw_class_on_image(label, img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
