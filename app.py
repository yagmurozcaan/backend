from flask import Flask, render_template, request
import os, csv, json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = "backend/uploads"
REPORT_FILE = "backend/reports/reports.csv"
MODEL_PATH = "Train/models/light_cnn_model.h5"  # Keras formatına çevrilmiş model

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Modeli yükle ----
model = load_model(MODEL_PATH)

# ---- ResNet50 Feature Extractor ----
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# ---- Kullanıcı giriş ekranı ----
@app.route('/')
def index():
    return render_template('index.html')

# ---- Kayıt ekranı ----
@app.route('/record', methods=['POST'])
def record():
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')
    return render_template('record.html', isim=isim, soyisim=soyisim)

# ---- Videodan kare çıkarma ve modele uygun feature çıkarma ----
def extract_features(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()

    # Video kısa ise pad et
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    # Feature extraction
    features_list = []
    for f in frames:
        img = image.img_to_array(f)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = resnet_model.predict(img, verbose=0)
        features_list.append(feat.flatten())  # 2048 boyutu

    features_array = np.array(features_list)  # (max_frames, 2048)
    return np.expand_dims(features_array, axis=0)  # (1, max_frames, 2048)

# ---- Video upload ve işleme ----
@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')

    if not video_file:
        return "Dosya bulunamadı", 400

    # MP4 olarak kaydet
    mp4_path = os.path.join(UPLOAD_FOLDER, video_file.filename.replace(".webm", ".mp4"))
    video_file.save(mp4_path)

    # Özellik çıkar
    features = extract_features(mp4_path)   # (1,30,2048)

    # Tahmin yap
    prediction = model.predict(features)

    # ---- Threshold ile karar ----
    threshold = 0.7  # Burayı istediğin değere ayarlayabilirsin
    if prediction.shape[-1] == 1:  # Sigmoid çıkış
        prob = prediction[0][0]
        pred_class = 1 if prob >= threshold else 0
    else:  # Softmax çıkış
        pred_class = int(np.argmax(prediction[0]))

    final_prediction = "olabilir" if pred_class == 1 else "değil"

    # CSV'ye kaydet
    with open(REPORT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["isim","soyisim","dosya","prediction_raw","final_prediction"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "isim": isim,
            "soyisim": soyisim,
            "dosya": os.path.basename(mp4_path),
            "prediction_raw": json.dumps(prediction.tolist()),
            "final_prediction": final_prediction
        })

    return f"Başarılı - Tahmin: {final_prediction}", 200

# ---- Admin giriş ekranı ----
@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# ---- Admin paneli ----
@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username != 'admin' or password != '1234':
            return "Hatalı kullanıcı adı veya şifre", 403

    if os.path.exists(REPORT_FILE):
        import pandas as pd
        df = pd.read_csv(REPORT_FILE)
        records = df.to_dict(orient='records')
    else:
        records = []
    return render_template('admin.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
