import os
import io
import time
import json
import base64
import sqlite3
import threading
import glob
import shutil
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

CUSTOM_MODEL_PATH = '/opt/smartpepper-yolo/models/best.pt'
BASE_MODEL_PATH = '/opt/smartpepper-yolo/models/plant_disease_base.pt'
FRUIT_MODEL_PATH = '/opt/smartpepper-yolo/pepper_fruit_model.pt'
COCO_MODEL_PATH = '/opt/smartpepper-yolo/yolov8n.pt'
DB_PATH = '/opt/smartpepper-yolo/data/farm.sqlite'

# Load model daun, model buah lada, dan model verifikasi non-tanaman
custom_model = YOLO(CUSTOM_MODEL_PATH)
base_model = YOLO(BASE_MODEL_PATH) if os.path.exists(BASE_MODEL_PATH) else None
fruit_model = YOLO(FRUIT_MODEL_PATH) if os.path.exists(FRUIT_MODEL_PATH) else None
coco_model = YOLO(COCO_MODEL_PATH) if os.path.exists(COCO_MODEL_PATH) else None

# Kamus terjemahan objek non-tanaman ke bahasa Indonesia sehari-hari
NON_PLANT_DICT = {
    'person': 'Manusia / Wajah / Tangan',
    'cell phone': 'Handphone / Gadget',
    'laptop': 'Layar Laptop / Monitor',
    'tv': 'Televisi / Monitor',
    'keyboard': 'Keyboard Komputer',
    'mouse': 'Mouse Komputer',
    'bottle': 'Botol Minum / Wadah',
    'cup': 'Cangkir / Gelas',
    'chair': 'Kursi / Mebel',
    'couch': 'Sofa / Tempat Duduk',
    'bed': 'Tempat Tidur',
    'dining table': 'Meja Makan / Perabot',
    'book': 'Buku / Kertas',
    'clock': 'Jam Dinding / Tangan',
    'scissors': 'Gunting / Logam',
    'backpack': 'Tas Ransel',
    'umbrella': 'Payung',
    'handbag': 'Tas Tangan',
    'tie': 'Dasi / Pakaian',
    'suitcase': 'Koper',
    'car': 'Mobil / Kendaraan',
    'motorcycle': 'Sepeda Motor',
    'bicycle': 'Sepeda',
    'cat': 'Kucing / Hewan',
    'dog': 'Anjing / Hewan'
}

DISEASE_KB = {
    "lada_sehat": {
        "name": "Daun Lada Sehat (Tumbuh Subur)",
        "condition": "SEHAT",
        "color": "#10b981", # hijau segar
        "pathogen": "Aman (Tidak Ada Penyakit)",
        "severity": "Bagus Sekali (Bebas Hama & Jamur)",
        "description": "Warna daun hijau segar merata, daun tidak layu, dan tidak ada bercak coklat atau hitam.",
        "pengendalian_hayati": "Campur pupuk kandang matang atau kompos dengan jamur baik (seperti Trichoderma) lalu taburkan di sekitar pangkal batang saat awal musim hujan agar tanah tetap sehat.",
        "pengendalian_kimiawi": "Cukup beri pupuk dasar yang berimbang. Belum perlu memakai obat kimia atau pestisida.",
        "teknik_budidaya": "Pastikan tiang panjat/penegak hidup tidak terlalu rimbun agar tanaman tetap terkena sinar matahari, dan buat saluran air agar air hujan tidak menggenang di bawah pohon.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "lada_tidak_sehat": {
        "name": "Penyakit Busuk Batang & Daun (Layu Mendadak)",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e", # merah bahaya
        "pathogen": "Jamur Busuk Pangkal Batang (Phytophthora) / Cacing Akar",
        "severity": "Gawat (Bisa Bikin Tanaman Mati Cepat)",
        "description": "Daun menguning kaku, layu tiba-tiba tapi daun masih menempel, cabang berguguran, atau pangkal batang bawah menghitam dan berlendir.",
        "pengendalian_hayati": "Siram pangkal batang dengan ramuan jamur baik Trichoderma atau bakteri pengurai (seperti Pseudomonas fluorescens / Natural GLIO) agar jamur jahat mati.",
        "pengendalian_kimiawi": "1) Siram tanah sekitar pohon dengan racikan Bubur Bordo (campuran 1 ons terusi biru + 1 ons kapur tembok dalam 5 liter air); 2) Untuk cacing/nematoda akar, taburkan sedikit insektisida butiran seperti karbofuran (seukuran 2 sendok makan per pohon) lalu tutup tanah.",
        "teknik_budidaya": "Jika tanaman sudah terlanjur mati atau busuk parah, segera cabut dan bakar di tempat agar tidak menular ke pohon lada di sebelahnya. Buat parit keliling agar air hujan tidak mengalir dari pohon sakit ke pohon sehat.",
        "actuator": {"fan": "HIGH (Airflow 100%)", "pump": "REDUCE", "growlight": "ON"}
    },
    "Pepper__bell___healthy": {
        "name": "Daun Lada Sehat",
        "condition": "SEHAT",
        "color": "#10b981",
        "pathogen": "Aman",
        "severity": "Bagus Sekali",
        "description": "Daun mulus dan hijau, tanaman lada tumbuh normal tanpa kendala.",
        "pengendalian_hayati": "Taburkan pupuk kompos yang dicampur jamur pengurai secara berkala.",
        "pengendalian_kimiawi": "Berikan pupuk tanaman secukupnya sesuai jadwal.",
        "teknik_budidaya": "Jaga tanah tetap gembur dan tidak becek.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "Pepper__bell___Bacterial_spot": {
        "name": "Bercak Daun & Jamur Lada",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e",
        "pathogen": "Bakteri / Jamur Daun",
        "severity": "Perlu Diobati",
        "description": "Ada bercak-bercak bintik hitam melingkar atau daun terbakar kecokelatan.",
        "pengendalian_hayati": "Semprot tanaman dengan bio-pestisida atau jamur baik Trichoderma di sore hari.",
        "pengendalian_kimiawi": "Semprotkan obat antijamur/fungisida tembaga atau Bubur Bordo, campurkan sedikit perekat (sabun perata) agar obat tidak luntur saat hujan.",
        "teknik_budidaya": "Pangkas daun-daun bawah yang menempel ke tanah dan bersihkan rumput liar di sekitar pohon.",
        "actuator": {"fan": "HIGH (100%)", "pump": "REDUCE", "growlight": "ON"}
    },
    "lada_buah_sehat": {
        "name": "Buah Lada Sehat (Butir Bersih)",
        "condition": "SEHAT",
        "color": "#10b981",
        "pathogen": "Aman (Buah Bagus)",
        "severity": "Kondisi Prima (Siap Panen)",
        "description": "Butiran buah lada bulat padat, kulit bersih mengkilap, tangkai buah segar, tidak ada bintik busuk.",
        "pengendalian_hayati": "Semprotkan pupuk organik cair yang mengandung mikroba baik saat mulai berbuah agar buah tahan penyakit.",
        "pengendalian_kimiawi": "Beri pupuk kalsium dan kalium agar kulit buah tebal, kuat, dan tidak gampang pecah.",
        "teknik_budidaya": "Pastikan buah mendapat sinar matahari pagi yang cukup dan angin bisa masuk leluasa.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "lada_buah_tidak_sehat": {
        "name": "Buah Lada Busuk / Hitam (Antraknosa)",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e",
        "pathogen": "Jamur Busuk Buah (Patek / Antraknosa)",
        "severity": "Gawat (Buah Cepat Rontok & Kering)",
        "description": "Buah lada berbercak cokelat kehitaman, keriput sebelum waktunya, tangkai buah membusuk dan rontok massal ke tanah.",
        "pengendalian_hayati": "Semprotkan jamur baik Trichoderma atau bakteri antagonis langsung ke dompolan buah lada.",
        "pengendalian_kimiawi": "Semprot dengan fungisida tembaga atau mankozeb, dan siramkan air kapur sirih atau Bubur Bordo ke tanah bila sudah mulai menular luas.",
        "teknik_budidaya": "Petik semua buah yang sudah busuk atau rontok lalu bakar jauh dari kebun. Kurangi rimbun daun pohon pelindung agar kebun tidak terlalu lembap dan gelap.",
        "actuator": {"fan": "HIGH (Airflow 100%)", "pump": "REDUCE", "growlight": "ON"}
    }
}

TRAIN_STATE = {
    "status": "IDLE",
    "epoch": 0,
    "total_epochs": 15,
    "progress": 0,
    "mAP50": 0.0,
    "loss": 0.0,
    "message": "Siap melatih dataset lada sehat vs tidak sehat."
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        temperature REAL, humidity REAL, soil_moisture REAL,
        water_level REAL, solar_voltage REAL, solar_current REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        condition TEXT,
        class_name TEXT,
        confidence REAL,
        source_type TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS actuators (
        id TEXT PRIMARY KEY, name TEXT, state INTEGER, mode TEXT
    )''')
    for act in [
        ('pump_irrigation', 'Submersible Pump 12V (Reservoir)', 0, 'AUTO'),
        ('growlight_spectrum', 'LED Grow Light Full Spectrum 50W', 1, 'AUTO'),
        ('fan_exhaust', 'Exhaust Fan Ventilasi 12V', 1, 'AUTO')
    ]:
        c.execute('INSERT OR IGNORE INTO actuators VALUES (?,?,?,?)', act)
    conn.commit()
    conn.close()

init_db()


def count_dataset():
    train_healthy = len(glob.glob('/opt/smartpepper-yolo/dataset/train/labels/*lada_sehat*.txt'))
    train_diseased = len(glob.glob('/opt/smartpepper-yolo/dataset/train/labels/*lada_tidak_sehat*.txt'))
    total_train = len(glob.glob('/opt/smartpepper-yolo/dataset/train/images/*'))
    total_val = len(glob.glob('/opt/smartpepper-yolo/dataset/val/images/*'))
    return {
        "train_healthy": train_healthy,
        "train_diseased": train_diseased,
        "total_train": total_train,
        "total_val": total_val
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    import random
    t = round(28.5 + random.uniform(-0.6, 0.8), 1)
    h = round(67.0 + random.uniform(-1.5, 2.5), 1)
    sm = round(72.0 + random.uniform(-1.0, 1.0), 1)
    wl = round(83.5 + random.uniform(-0.3, 0.3), 1)
    sv = round(12.8 + random.uniform(-0.1, 0.3), 2)
    sc = round(1.85 + random.uniform(-0.05, 0.1), 2)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO telemetry (temperature, humidity, soil_moisture, water_level, solar_voltage, solar_current)
                 VALUES (?,?,?,?,?,?)''', (t, h, sm, wl, sv, sc))
    c.execute('SELECT id, name, state, mode FROM actuators')
    acts = {row[0]: {'name': row[1], 'state': row[2], 'mode': row[3]} for row in c.fetchall()}
    conn.commit()
    conn.close()

    return jsonify({
        'temperature': t, 'humidity': h, 'soil_moisture': sm,
        'water_level': wl, 'solar_voltage': sv, 'solar_current': sc,
        'actuators': acts, 'timestamp': time.strftime('%H:%M:%S')
    })

@app.route('/api/actuators/toggle', methods=['POST'])
def toggle_actuator():
    data = request.json or {}
    act_id = data.get('id')
    new_state = int(data.get('state', 0))
    mode = data.get('mode', 'MANUAL')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE actuators SET state = ?, mode = ? WHERE id = ?', (new_state, mode, act_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'SUCCESS', 'id': act_id, 'state': new_state, 'mode': mode})

def check_is_non_plant(img_np, conf_thresh=0.30):
    h, w, _ = img_np.shape
    detected_objects = []

    # A. Deteksi objek sehari-hari non-tanaman lewat COCO YOLOv8
    if coco_model is not None:
        res_coco = coco_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res_coco.boxes:
            c_name = coco_model.names.get(int(box.cls[0]), '')
            if c_name in NON_PLANT_DICT:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detected_objects.append({
                    'class_raw': c_name,
                    'name_id': NON_PLANT_DICT[c_name],
                    'confidence': round(conf * 100, 1),
                    'bbox': [round(x, 1) for x in xyxy],
                    'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)]
                })

    # B. Analisis Rasio Spektrum Warna Tanaman (Klorofil Hijau, Klorosis Kuning/Coklat, Buah Matang)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (25, 25, 25), (90, 255, 255))
    mask_yellow_brown = cv2.inRange(hsv, (10, 35, 25), (25, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    plant_mask = mask_green | mask_yellow_brown | mask_red1 | mask_red2
    plant_pixel_count = cv2.countNonZero(plant_mask)
    plant_ratio = round((plant_pixel_count / (h * w)) * 100, 1)

    return detected_objects, plant_ratio

def perform_ensemble_detection(img_np, conf_thresh=0.35, target='all'):
    h, w, _ = img_np.shape
    detections = []
    
    # 1. Jalankan Model Buah jika target 'all' atau 'fruit'
    if target in ('all', 'fruit') and fruit_model is not None:
        res_fruit = fruit_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res_fruit.boxes:
            cls_id = int(box.cls[0])
            cls_name = fruit_model.names.get(cls_id, 'lada_buah_sehat')
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'source': 'yolo_fruit_model',
                'organ': 'BUAH',
                'class_raw': cls_name,
                'confidence': round(conf * 100, 1),
                'bbox': [round(x, 1) for x in xyxy],
                'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)],
                'condition': 'SEHAT' if 'sehat' in cls_name and 'tidak' not in cls_name else 'TIDAK SEHAT'
            })

    # 2. Deteksi Daun Menggunakan Base Model Plant Disease (Sangat Akurat & Kebal False Positive pada Kulit/Dinding)
    if target in ('all', 'leaf') and base_model is not None:
        res_base = base_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res_base.boxes:
            cls_id = int(box.cls[0])
            cls_name = base_model.names.get(cls_id, '')
            if 'pepper' in cls_name.lower() or 'healthy' in cls_name.lower():
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cond = 'SEHAT' if 'healthy' in cls_name.lower() else 'TIDAK SEHAT'
                
                # Verifikasi warna di area bounding box harus mengandung klorofil hijau daun atau nekrosis daun lada
                bx0, by0, bx1, by1 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(w, int(xyxy[2])), min(h, int(xyxy[3]))
                crop = img_np[by0:by1, bx0:bx1]
                if crop.size > 0:
                    c_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                    c_green = cv2.inRange(c_hsv, (28, 25, 25), (88, 255, 255))
                    c_ratio = cv2.countNonZero(c_green) / (crop.shape[0] * crop.shape[1])
                    if c_ratio < 0.05 and 'healthy' in cls_name.lower():
                        continue # Abaikan bila diklaim healthy tapi tidak ada klorofil daun
                
                detections.append({
                    'source': 'plant_pathology_base',
                    'organ': 'DAUN',
                    'class_raw': 'lada_sehat' if cond == 'SEHAT' else 'lada_tidak_sehat',
                    'confidence': round(conf * 100, 1),
                    'bbox': [round(x, 1) for x in xyxy],
                    'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)],
                    'condition': cond
                })

    return detections

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    file = request.files.get('image')
    conf_thresh = float(request.form.get('conf', 0.25))
    target = request.form.get('target', 'all') # 'all', 'leaf', 'fruit'
    if not file:
        return jsonify({'error': 'Tidak ada berkas citra'}), 400

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    h, w, _ = img_np.shape

    # 1. Verifikasi Non-Tanaman (Bukan Daun / Bukan Buah)
    non_plant_objs, plant_color_ratio = check_is_non_plant(img_np, conf_thresh=0.30)
    
    # Cek klorofil hijau khusus daun
    hsv_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    green_pixels = cv2.countNonZero(cv2.inRange(hsv_full, (26, 25, 25), (88, 255, 255)))
    green_ratio = round((green_pixels / (h * w)) * 100, 1)

    # Objek tergolong BUKAN DAUN bila terdeteksi COCO object (orang, gadget, perabot) ATAU tidak ada warna daun/buah yang memadai
    is_non_plant = (len(non_plant_objs) > 0 and green_ratio < 10.0) or (target == 'leaf' and green_ratio < 2.5) or (plant_color_ratio < 2.0)

    if is_non_plant:
        annotated_img = img_np.copy()
        identified_labels = []
        for obj in non_plant_objs:
            box = [int(v) for v in obj['bbox']]
            cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), (234, 179, 8), 2)
            cv2.putText(annotated_img, f"{obj['name_id']} {obj['confidence']}%", (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (234, 179, 8), 2)
            identified_labels.append(obj['name_id'])

        pil_ann = Image.fromarray(annotated_img)
        buf = io.BytesIO()
        pil_ann.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        desc = f"Terdeteksi objek bukan daun: {', '.join(set(identified_labels))}" if identified_labels else "Citra tidak mengandung ciri visual daun atau buah lada."

        return jsonify({
            'status': 'SUCCESS',
            'overall_condition': 'BUKAN_DAUN',
            'count': 0,
            'non_plant_detected': True,
            'plant_ratio': plant_color_ratio,
            'non_plant_objects': non_plant_objs,
            'message': 'Foto yang diunggah terdeteksi BUKAN daun/buah tanaman lada.',
            'detections': [{
                'organ': 'BUKAN_TANAMAN',
                'condition': 'BUKAN DAUN / TANAMAN',
                'confidence': non_plant_objs[0]['confidence'] if non_plant_objs else 95.0,
                'class_raw': non_plant_objs[0]['class_raw'] if non_plant_objs else 'bukan_daun',
                'norm_bbox': non_plant_objs[0]['norm_bbox'] if non_plant_objs else [0.1, 0.1, 0.9, 0.9],
                'info': {
                    'name': 'Objek Terdeteksi Bukan Daun Lada',
                    'condition': 'BUKAN DAUN',
                    'color': '#eab308',
                    'pathogen': 'Bukan Tanaman',
                    'severity': 'Salah Sampel',
                    'description': desc + f" (Kandungan pigmen tanaman hanya {plant_color_ratio}%).",
                    'pengendalian_hayati': 'Harap arahkan kamera atau foto langsung ke helai daun atau butiran buah lada asli.',
                    'pengendalian_kimiawi': 'Tidak ada tindakan kimiawi yang diperlukan.',
                    'teknik_budidaya': 'Pastikan daun atau buah lada berada di jarak 15-30 cm agar kamera fokus.'
                }
            }],
            'annotated_image': f'data:image/jpeg;base64,{b64}'
        })

    # 2. Jika lolos verifikasi tanaman, jalankan inferensi ensemble daun & buah lada
    detections = perform_ensemble_detection(img_np, conf_thresh, target)

    # Render bounding boxes di atas gambar
    annotated_img = img_np.copy()
    for d in detections:
        box = [int(v) for v in d['bbox']]
        is_healthy = d['condition'] == 'SEHAT'
        color = (16, 185, 129) if is_healthy else (244, 63, 94) # RGB (Emerald vs Rose)
        cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), color, max(2, int(w/250)))
        organ_tag = f"{d.get('organ', '')} " if d.get('organ') else ""
        label = f"{organ_tag}{d['condition']} {d['confidence']}%"
        cv2.putText(annotated_img, label, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_img, label, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        fallback_key = ('lada_buah_sehat' if is_healthy else 'lada_buah_tidak_sehat') if d.get('organ') == 'BUAH' else ('lada_sehat' if is_healthy else 'lada_tidak_sehat')
        info = DISEASE_KB.get(d['class_raw'], DISEASE_KB[fallback_key])
        d['info'] = info

    # Base64 output
    pil_ann = Image.fromarray(annotated_img)
    buf = io.BytesIO()
    pil_ann.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Catat log deteksi
    if detections:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for d in detections:
            c.execute('INSERT INTO detections (condition, class_name, confidence, source_type) VALUES (?,?,?,?)',
                      (d['condition'], d['class_raw'], d['confidence'], 'REALTIME_INFERENCE'))
        conn.commit()
        conn.close()

    if len(detections) == 0:
        overall_condition = "TIDAK_ADA_SAMPEL"
    elif all(d['condition'] == 'SEHAT' for d in detections):
        overall_condition = "SEHAT"
    else:
        overall_condition = "TIDAK SEHAT" 

    return jsonify({
        'status': 'SUCCESS',
        'overall_condition': overall_condition,
        'count': len(detections),
        'detections': detections,
        'annotated_image': f'data:image/jpeg;base64,{b64}'
    })

# Endpoint Dataset & Training

from flask import send_from_directory

@app.route('/api/dataset/gallery', methods=['GET'])
def get_dataset_gallery():
    split = request.args.get('split', 'train')
    category = request.args.get('category', 'all') # 'all', 'lada_sehat', 'lada_tidak_sehat'
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 24))

    img_dir = f'/opt/smartpepper-yolo/dataset/{split}/images'
    all_files = sorted(glob.glob(f'{img_dir}/*'), key=os.path.getmtime, reverse=True)
    
    filtered = []
    for fpath in all_files:
        base = os.path.basename(fpath)
        is_healthy = 'lada_sehat' in base
        is_diseased = 'lada_tidak_sehat' in base
        if category == 'lada_sehat' and not is_healthy:
            continue
        if category == 'lada_tidak_sehat' and not is_diseased:
            continue
        
        filtered.append({
            'filename': base,
            'split': split,
            'category': 'lada_sehat' if is_healthy else ('lada_tidak_sehat' if is_diseased else 'unknown'),
            'label_title': 'Lada Sehat' if is_healthy else 'Lada Tidak Sehat',
            'url': f'/api/dataset/image/{split}/{base}'
        })

    total_items = len(filtered)
    total_pages = max(1, (total_items + per_page - 1) // per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    return jsonify({
        'total': total_items,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'items': filtered[start_idx:end_idx]
    })

@app.route('/api/dataset/image/<split>/<filename>')
def serve_dataset_image(split, filename):
    safe_split = 'val' if split == 'val' else 'train'
    img_dir = f'/opt/smartpepper-yolo/dataset/{safe_split}/images'
    return send_from_directory(img_dir, filename)

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    stats = count_dataset()
    stats["training_state"] = TRAIN_STATE
    return jsonify(stats)

@app.route('/api/dataset/add-sample', methods=['POST'])
def add_sample():
    file = request.files.get('image')
    label = request.form.get('label')
    split = request.form.get('split', 'train')

    if not file or label not in ['0', '1', 'lada_sehat', 'lada_tidak_sehat']:
        return jsonify({'error': 'Parameter tidak valid'}), 400

    cls_id = 0 if label in ['0', 'lada_sehat'] else 1
    cls_tag = 'lada_sehat' if cls_id == 0 else 'lada_tidak_sehat'

    ts = int(time.time() * 1000)
    base_name = f"sample_{cls_tag}_{ts}"
    img_ext = os.path.splitext(file.filename)[1].lower() or '.jpg'
    img_filename = f"{base_name}{img_ext}"
    label_filename = f"{base_name}.txt"

    img_path = f"/opt/smartpepper-yolo/dataset/{split}/images/{img_filename}"
    lbl_path = f"/opt/smartpepper-yolo/dataset/{split}/labels/{label_filename}"

    file.save(img_path)
    with open(lbl_path, "w") as f:
        f.write(str(cls_id) + " 0.5 0.5 0.85 0.85\n")

    return jsonify({
        'status': 'SUCCESS',
        'message': f'Sampel {cls_tag} tersimpan di {split}.',
        'stats': count_dataset()
    })

def prepare_val_dataset():
    val_imgs = glob.glob('/opt/smartpepper-yolo/dataset/val/images/*')
    train_imgs = glob.glob('/opt/smartpepper-yolo/dataset/train/images/*')
    if len(val_imgs) == 0 and len(train_imgs) > 0:
        for t_img in train_imgs:
            base = os.path.basename(t_img)
            name_no_ext = os.path.splitext(base)[0]
            shutil.copy(t_img, f'/opt/smartpepper-yolo/dataset/val/images/{base}')
            lbl_file = f'/opt/smartpepper-yolo/dataset/train/labels/{name_no_ext}.txt'
            if os.path.exists(lbl_file):
                shutil.copy(lbl_file, f'/opt/smartpepper-yolo/dataset/val/labels/{name_no_ext}.txt')

def run_training_worker(epochs=15, imgsz=224):
    global TRAIN_STATE, custom_model
    try:
        prepare_val_dataset()
        TRAIN_STATE["status"] = "TRAINING"
        TRAIN_STATE["total_epochs"] = epochs
        TRAIN_STATE["progress"] = 5
        TRAIN_STATE["message"] = "Inisialisasi transfer learning YOLOv8..."

        def on_fit_epoch_end(trainer):
            ep = trainer.epoch + 1
            TRAIN_STATE["epoch"] = ep
            TRAIN_STATE["progress"] = int((ep / epochs) * 90) + 5
            TRAIN_STATE["message"] = f"Epoch {ep}/{epochs} selesai."

        train_model = YOLO('yolov8n.pt')
        train_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        train_model.train(
            data='/opt/smartpepper-yolo/dataset/data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=4,
            device='cpu',
            project='/opt/smartpepper-yolo/runs',
            name='pepper_train',
            exist_ok=True,
            verbose=False
        )

        best_weight = '/opt/smartpepper-yolo/runs/pepper_train/weights/best.pt'
        if os.path.exists(best_weight):
            shutil.copyfile(best_weight, '/opt/smartpepper-yolo/models/best.pt')
            custom_model = YOLO('/opt/smartpepper-yolo/models/best.pt')

        TRAIN_STATE["status"] = "COMPLETED"
        TRAIN_STATE["progress"] = 100
        TRAIN_STATE["message"] = "Training selesai! Model lada sehat vs tidak sehat aktif digunakan."

    except Exception as e:
        TRAIN_STATE["status"] = "FAILED"
        TRAIN_STATE["message"] = f"Training gagal: {str(e)}"

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global TRAIN_STATE
    if TRAIN_STATE["status"] == "TRAINING":
        return jsonify({'error': 'Training sedang berlangsung'}), 400

    stats = count_dataset()
    if stats["total_train"] < 2:
        return jsonify({'error': 'Minimal kumpulkan 2 sampel untuk training.'}), 400

    data = request.json or {}
    epochs = int(data.get('epochs', 15))

    t = threading.Thread(target=run_training_worker, kwargs={'epochs': epochs})
    t.daemon = True
    t.start()
    return jsonify({'status': 'SUCCESS', 'message': f'Training {epochs} epoch dimulai.'})

def generate_video_stream():
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 24, 28)
        cv2.rectangle(frame, (100, 220), (280, 420), (35, 45, 55), -1)
        cv2.rectangle(frame, (360, 220), (540, 420), (35, 45, 55), -1)
        cv2.rectangle(frame, (100, 220), (280, 420), (0, 200, 120), 2)
        cv2.rectangle(frame, (360, 220), (540, 420), (0, 200, 120), 2)
        cv2.circle(frame, (190, 300), 65, (40, 140, 50), -1)
        cv2.circle(frame, (450, 300), 65, (40, 140, 50), -1)
        cv2.putText(frame, "PEPPER POT #1", (125, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "PEPPER POT #2", (385, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SMARTPEPPER 4.0 - TOP CANOPY CV CAMERA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"FPS: 25.0 | {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/api/camera/stream')
def camera_stream():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
