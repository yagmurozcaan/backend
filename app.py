from flask import Flask, render_template, request
import os, csv
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# -------------------
# Ayarlar
# -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORT_FILE = "reports/reports.csv"
#MODEL_PATH = "models/best_model_20251002_223638.keras"
MODEL_PATH = "models/best_model_20251005_131650.keras"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

# -------------------
# Model ve Backbone
# -------------------
print("✓ Model yükleniyor...")
model = load_model(MODEL_PATH)
print("✓ EfficientNetB0 yükleniyor...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

# -------------------
# Parametreler
# -------------------
T = 32                 # Eğitimde kullanılan kare sayısı
n_csv_features = 7     # Eğitimde kullanılan CSV feature sayısı
THRESHOLD = 0.70      # Threshold (modeline göre değiştir)

# -------------------
# Videodan feature çıkarma
# -------------------
def extract_features(video_path, max_frames=T):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    # Kareleri oku
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()

  

    # EfficientNet feature çıkar
    features_list = []
    for f in frames:
        img = image.img_to_array(f)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = base_model.predict(img, verbose=0)
        features_list.append(feat.flatten())

    features_array = np.array(features_list)  # (T,1280)

    # CSV feature ekle (dummy zeros)
    csv_features = np.zeros((features_array.shape[0], n_csv_features))
    features_array = np.concatenate([features_array, csv_features], axis=1)  # (T,1287)

    return np.expand_dims(features_array, axis=0)  # (1,T,1287)


# -------------------
#  giriş ekranı
# -------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------
# Kullanıcı giriş ekranı
# -------------------
@app.route('/index')
def index():
    return render_template('index.html')

# -------------------
# Kayıt ekranı
# -------------------
@app.route('/record', methods=['POST'])
def record():
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')
    return render_template('record.html', isim=isim, soyisim=soyisim)

# -------------------
# Video upload ve işleme
# -------------------
@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')

    if not video_file:
        return "Dosya bulunamadı", 400

    # MP4 olarak kaydet
    mp4_path = os.path.join(UPLOAD_FOLDER, video_file.filename.rsplit(".", 1)[0] + ".mp4")
    video_file.save(mp4_path)

    # Özellik çıkar
    try:
        features = extract_features(mp4_path)   # (1, T, 1287)
    except Exception as e:
        return f"Video işlenemedi: {str(e)}", 500

    # Tahmin yap
    prediction = model.predict(features, verbose=0)
    prob = float(prediction[0][0])
    pred_class = 1 if prob >= THRESHOLD else 0
    final_prediction = "Otizm olabilir" if pred_class == 1 else "Otizm değil"

    # CSV'ye kaydet
    with open(REPORT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["isim","soyisim","dosya","probability","final_prediction"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "isim": isim,
            "soyisim": soyisim,
            "dosya": os.path.basename(mp4_path),
            "probability": float(prob),
            "final_prediction": final_prediction
        })

    return f"Başarılı ✅ - Tahmin: {final_prediction} (olasılık: {prob:.2f})", 200

# -------------------
# Admin giriş ekranı
# -------------------
@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# -------------------
# Admin paneli
# -------------------
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


# -------------------
# kapanış ekranı
# -------------------
@app.route('/finish')
def finish():
    return render_template('finish.html')

# -------------------
# Run
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
