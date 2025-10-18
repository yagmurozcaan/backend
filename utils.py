import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp
import os
import sqlite3

DB_FILE = "reports/reports.db"

# --- Landmark feature extraction ---
def extract_landmark_features(video_path, max_frames=32):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    hand_motion_values = []
    head_motion_values = []
    direction_changes = 0
    prev_left_hand = None
    prev_direction = None
    prev_head_angle = None
    blink_count = 0
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

            # Body height normalization
            body_height = 1.0
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                h = abs(shoulder.y - hip.y)
                body_height = h if h > 1e-6 else 1.0

            # Hand motion
            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    if prev_left_hand is not None:
                        dx = abs(wrist.x - prev_left_hand[0]) / body_height
                        dy = abs(wrist.y - prev_left_hand[1]) / body_height
                        motion = dx + dy
                        hand_motion_values.append(motion)

                        direction = np.sign(wrist.y - prev_left_hand[1])
                        if prev_direction is not None and direction != prev_direction:
                            direction_changes += 1
                        prev_direction = direction
                    prev_left_hand = (wrist.x, wrist.y)

            # Head motion
            if face_res.multi_face_landmarks:
                mesh = face_res.multi_face_landmarks[0]
                left_eye = mesh.landmark[33]
                right_eye = mesh.landmark[263]
                dx = right_eye.x - left_eye.x
                dy = right_eye.y - left_eye.y
                head_angle = np.degrees(np.arctan2(dy, dx))
                if prev_head_angle is not None:
                    head_motion_values.append(abs(head_angle - prev_head_angle))
                prev_head_angle = head_angle

                # Blink detection
                left_eye_top = mesh.landmark[159]
                left_eye_bottom = mesh.landmark[145]
                eye_dist = abs(left_eye_top.y - left_eye_bottom.y)
                if prev_eye_dist is not None and eye_dist < 0.02 <= prev_eye_dist:
                    blink_count += 1
                prev_eye_dist = eye_dist

            frame_count += 1

    cap.release()

    hand_motion = np.mean(hand_motion_values) if hand_motion_values else 0
    head_motion = np.mean(head_motion_values) if head_motion_values else 0

    hand_threshold = (np.mean(hand_motion_values) + np.std(hand_motion_values)) if hand_motion_values else 0.02
    head_threshold = (np.mean(head_motion_values) + np.std(head_motion_values)) if head_motion_values else 0.015

    armflapping = int(hand_motion > hand_threshold or direction_changes > 3)
    headbanging = int(head_motion > head_threshold)
    spinning = int(hand_motion > hand_threshold * 1.5 and head_motion > head_threshold)
    blink = int(blink_count > 0)

    return np.array([armflapping, headbanging, spinning, blink], dtype=float)

# --- Features extraction per segment ---
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

# --- DB kaydı ---
def save_report_and_segments(video_path, final_prediction, avg_prob, segments_info, isim="isim", soyisim="soyisim"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    video_name = os.path.basename(video_path)

    # Ana tabloya segmentlerden gelen landmark değerlerini kaydet
    if segments_info:
        last_landmarks = segments_info[-1]["landmarks"]
        arm, head, spin, blink = last_landmarks
    else:
        arm = head = spin = blink = 0

    # reports tablosuna genel rapor
    c.execute("""
        INSERT INTO reports 
        (isim, soyisim, dosya, probability, final_prediction, armflapping, headbanging, spinning, blink)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (isim, soyisim, video_name, float(avg_prob), final_prediction, int(arm), int(head), int(spin), int(blink)))
    report_id = c.lastrowid

    # segment_outliers tablosuna segmentler
    for idx, seg in enumerate(segments_info):
        arm, head, spin, blink = seg["landmarks"]
        c.execute("""
            INSERT INTO segment_outliers
            (report_id, segment_index, probability, armflapping, headbanging, spinning, blink)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (report_id, idx, float(seg["prob"]), int(arm), int(head), int(spin), int(blink)))

    conn.commit()
    conn.close()
    print(f"✅ Video ve segmentler veritabanına kaydedildi: {video_name}")
    return report_id

# --- Video prediction ---
def predict_video_with_segments(video_path, model, base_model, segment_length_sec=2, fps=30, threshold=0.46, isim="isim", soyisim="soyisim"):
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
            "segment_index": i,
            "start": start_sec,
            "end": end_sec,
            "prob": pred_prob,
            "landmarks": landmark_segments[i]
        })

    avg_prob = float(np.mean(probs))
    outlier_segments = [seg for seg in segments_info if abs(seg["prob"] - avg_prob) > 0.15]
    filtered_probs = [seg["prob"] for seg in segments_info if seg not in outlier_segments]
    if filtered_probs:
        avg_prob = float(np.mean(filtered_probs))

    final_class = int(avg_prob >= threshold)
    final_prediction = "Otizm olabilir" if final_class else "Otizm değil"

    # Veritabanına kaydet
    save_report_and_segments(video_path, final_prediction, avg_prob, segments_info, isim, soyisim)

    return final_prediction, avg_prob, segments_info, outlier_segments
