from flask import Flask, render_template, request
import os, sqlite3, csv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from init_db import init_db
from utils import extract_features_segments, predict_video_with_segments

# ------------------- Flask Ayarları -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DB_FILE = "reports/reports.db"
#MODEL_PATH = "models/best_model_20251014_010814.keras"threshold 46-47
#MODEL_PATH = "best_model_20251014_194635.keras"#threshold 85
MODEL_PATH = "best_model_20251014_200636.keras"#threshold 70


SAPMA_FILE = "reports/outliers.csv"

THRESHOLD = 0.85

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(SAPMA_FILE), exist_ok=True)

# ------------------- DB init -------------------
if not os.path.exists(DB_FILE):
    init_db()

# ------------------- Model -------------------
model = load_model(MODEL_PATH)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False


# ------------------- Upload -------------------
@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')

    if not video_file:
        return "❌ Video dosyası bulunamadı", 400

    mp4_path = os.path.join(UPLOAD_FOLDER, video_file.filename.rsplit(".", 1)[0] + ".mp4")
    video_file.save(mp4_path)

    # prediction
    final_pred, avg_prob, segments_info, outlier_segments = predict_video_with_segments(
        mp4_path, model, base_model, threshold=THRESHOLD
    )

    # CSV kaydet
    with open(SAPMA_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ["isim","soyisim","dosya","segment_start","segment_end",
                    "probability","final_prediction","Hand Motion","Head Motion","Spinning","Blink"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for seg in segments_info:
            lm = seg.get("landmarks", [0,0,0,0])
            writer.writerow({
                "isim": isim,
                "soyisim": soyisim,
                "dosya": os.path.basename(mp4_path),
                "segment_start": seg["start"],
                "segment_end": seg["end"],
                "probability": seg["prob"],
                "final_prediction": final_pred,
                "Hand Motion": int(lm[0]),
                "Head Motion": int(lm[1]),
                "Spinning": int(lm[2]),
                "Blink": int(lm[3])
            })

    # DB kaydet (son landmark)
    final_landmarks = segments_info[-1].get("landmarks", [0,0,0,0])
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO reports (isim, soyisim, dosya, probability, final_prediction, armflapping, headbanging, spinning, blink)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        isim, soyisim, os.path.basename(mp4_path), avg_prob, final_pred,
        int(final_landmarks[0]), int(final_landmarks[1]), int(final_landmarks[2]), int(final_landmarks[3])
    ))
    conn.commit()
    conn.close()

    return f"✅ Tahmin: {final_pred} (olasılık: {avg_prob:.2f}), sapmalar CSV’ye kaydedildi", 200

# ------------------- Admin -------------------
@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM reports ORDER BY created_at DESC")
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('admin.html', records=records)

# ------------------- Giriş ekranı -------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')
    return render_template('record.html', isim=isim, soyisim=soyisim)

@app.route('/finish')
def finish():
    return render_template('finish.html')

# ------------------- Run -------------------
if __name__ == '__main__':
    app.run(debug=True)
