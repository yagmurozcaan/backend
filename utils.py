"""
Utility Functions for NEUROLOOK Project
Contains video processing, landmark extraction, and prediction functions.
Handles MediaPipe landmark detection, EfficientNet feature extraction, and LSTM model predictions.
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp

def extract_landmark_features(video_path, max_frames=32):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    hand_motion = 0
    head_motion = 0
    blink_count = 0
    prev_left_hand = None
    prev_head_y = None
    prev_eye_dist = None

    with mp_pose.Pose(static_image_mode=False) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
         mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face:

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(frame_rgb)
            hand_res = hands.process(frame_rgb)
            face_res = face.process(frame_rgb)

            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    if prev_left_hand is not None:
                        hand_motion += abs(wrist.y - prev_left_hand)
                    prev_left_hand = wrist.y

            if pose_res.pose_landmarks:
                nose = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                if prev_head_y is not None:
                    head_motion += abs(nose.y - prev_head_y)
                prev_head_y = nose.y

            if face_res.multi_face_landmarks:
                mesh = face_res.multi_face_landmarks[0]
                left_eye_top = mesh.landmark[159]
                left_eye_bottom = mesh.landmark[145]
                eye_dist = abs(left_eye_top.y - left_eye_bottom.y)
                if prev_eye_dist is not None and eye_dist < 0.02 and prev_eye_dist >= 0.02:
                    blink_count += 1
                prev_eye_dist = eye_dist

            frame_count += 1

    cap.release()
    frame_count = max(frame_count, 1)
    hand_motion /= frame_count
    head_motion /= frame_count

    armflapping = 1 if hand_motion > 0.02 else 0
    headbanging = 1 if head_motion > 0.015 else 0
    spinning = 1 if hand_motion > 0.03 and head_motion > 0.015 else 0
    blink = 1 if blink_count > 0 else 0

    return np.array([armflapping, headbanging, spinning, blink], dtype=float)

def extract_features_segments(video_path, base_model, segment_length_sec=2, fps=30, max_frames_per_segment=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frame_count = int(segment_length_sec * fps)
    features_segments, landmark_segments = [], []

    start_frame = 0
    while start_frame < total_frames:
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        count = 0
        while count < min(segment_frame_count, max_frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
        if not frames:
            break

        features_list = []
        for f in frames:
            img = image.img_to_array(f)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = base_model.predict(img, verbose=0)
            features_list.append(feat.flatten())
        features_array = np.array(features_list)

        landmark_features = extract_landmark_features(video_path, max_frames=len(frames))
        landmark_tiled = np.tile(landmark_features, (features_array.shape[0], 1))
        combined_features = np.concatenate([features_array, landmark_tiled], axis=1)
        features_segments.append(np.expand_dims(combined_features, axis=0))
        landmark_segments.append(landmark_features)

        start_frame += segment_frame_count

    cap.release()
    return features_segments, landmark_segments

def predict_video_with_segments(video_path, model, base_model, segment_length_sec=2, fps=30, threshold=0.46):
    features_segments, landmark_segments = extract_features_segments(video_path, base_model, segment_length_sec, fps)
    if not features_segments:
        return None, None, None, None

    probs, segments_info = [], []
    for i, seg_features in enumerate(features_segments):
        pred_prob = float(model.predict(seg_features, verbose=0)[0][0])
        probs.append(pred_prob)
        start_sec = i * segment_length_sec
        end_sec = start_sec + segment_length_sec
        segments_info.append({
            "start": start_sec,
            "end": end_sec,
            "prob": pred_prob,
            "landmarks": landmark_segments[i]  
        })

    avg_prob = np.mean(probs)
    final_class = 1 if avg_prob >= threshold else 0
    final_prediction = "Otizm olabilir" if final_class else "Otizm değil"

    outlier_segments = [seg for seg in segments_info if abs(seg["prob"] - avg_prob) > 0.15]

    return final_prediction, avg_prob, segments_info, outlier_segments
