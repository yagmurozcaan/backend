"""from flask import Flask, render_template, request, redirect, url_for
import os, csv
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REPORT_FILE = 'reports/rapor.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Ana giriş ekranı
@app.route('/')
def index():
    return render_template('index.html')

# Kullanıcı bilgisi sonrası kamera ekranı
@app.route('/record', methods=['POST'])
def record():
    isim = request.form.get("isim")
    soyisim = request.form.get("soyisim")
    return render_template('record.html', isim=isim, soyisim=soyisim)

# Video yükleme
@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')

    if video_file:
        save_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(save_path)

        # CSV'ye kayıt
        with open(REPORT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["isim", "soyisim", "dosya", "tarih"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "isim": isim,
                "soyisim": soyisim,
                "dosya": video_file.filename,
                "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return "Başarılı", 200
    return "Dosya bulunamadı", 400

# Admin giriş sayfası
@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# Admin doğrulama
@app.route('/admin', methods=['POST'])
def admin():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == "admin" and password == "1234":  # basit giriş kontrolü
        raporlar = []
        if os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                raporlar = list(reader)
        return render_template("admin.html", raporlar=raporlar)
    else:
        return "Yetkisiz giriş", 403

if __name__ == '__main__':
    app.run(debug=True)
"""
from flask import Flask, render_template, request
import os, csv, json
from moviepy.editor import VideoFileClip
import time

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORT_FILE = "reports.csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# ---- Video upload ve işleme ----
@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    isim = request.form.get('isim')
    soyisim = request.form.get('soyisim')

    if not video_file:
        return "Dosya bulunamadı", 400

    # Versiyonlu isim: kullanıcı adı + timestamp
    base_filename = f"{isim}_{soyisim}_{int(time.time())}"
    mp4_filename = f"{base_filename}.mp4"
    mp4_path = os.path.join(UPLOAD_FOLDER, mp4_filename)

    # Kaydet
    video_file.save(mp4_path)

    # ----- SEGMENTLERİ MODEL İÇİN HAZIRLA -----
    # Örnek: segment tahmini
    segment_predictions = [0, 1, 0, 1]
    final_prediction = "olabilir" if sum(segment_predictions)/len(segment_predictions) > 0.5 else "değil"

    # ----- CSV'YE KAYDET -----
    fieldnames = ["isim","soyisim","mp4_dosya","segmentler","final_prediction"]
    file_exists = os.path.exists(REPORT_FILE)
    with open(REPORT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "isim": isim,
            "soyisim": soyisim,
            "mp4_dosya": mp4_filename,
            "segmentler": json.dumps(segment_predictions),
            "final_prediction": final_prediction
        })

    return "Başarılı", 200

# ---- Admin giriş ekranı ----
@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# ---- Admin paneli ----
@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    import pandas as pd

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username != 'admin' or password != '1234':
            return "Hatalı kullanıcı adı veya şifre", 403

    if os.path.exists(REPORT_FILE):
        df = pd.read_csv(REPORT_FILE)
        records = df.to_dict(orient='records')
    else:
        records = []

    return render_template('admin.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
