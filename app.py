import os
import io
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from celery import Celery
from upscale import upscale
from celery.result import AsyncResult
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Конфигурация
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
PROCESSED_FOLDER = os.environ.get('PROCESSED_FOLDER', 'processed')
MODEL_PATH = os.environ.get('MODEL_PATH', 'EDSR_x2.pb')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(app.name, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Celery задача на апскейлинг
@celery.task(bind=True)
def upscale_task(self, image_bytes: bytes) -> str:
    # Из байт в numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")

    # Апскейлим
    result = upscale(image)

    # Из numpy array обратно в байты jpg/jpeg
    success, encoded_img = cv2.imencode('.jpg', result)
    if not success:
        raise RuntimeError("Failed to encode image")

    # Сохраняем в файл с именем task_id.jpg
    filename = f"{self.request.id}.jpg"
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    with open(filepath, 'wb') as f:
        f.write(encoded_img.tobytes())

    return filename  # возвращаем имя файла

# Роуты

# 1) POST /upscale - загрузить файл и запустить апскейлинг
@app.route('/upscale', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(image_file.filename):
        return jsonify({'error': 'File format not allowed'}), 400

    image_bytes = image_file.read()

    # Запускаем celery задачу с байтами
    task = upscale_task.apply_async(args=[image_bytes])

    return jsonify({'task_id': task.id, 'status': 'processing'}), 202

# 2) GET /tasks/<task_id> - получить статус задачи и ссылку на результат
@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {'task_id': task_id, 'status': task.status}

    if task.status == 'SUCCESS':
        filename = task.result  # имя файла, возвращённое задачей
        file_url = f"/processed/{filename}"
        response['result_url'] = file_url

    elif task.status == 'FAILURE':
        response['error'] = str(task.result)

    return jsonify(response)

# 3) GET /processed/<filename> - отдача обработанного файла
@app.route('/processed/<filename>', methods=['GET'])
def get_processed_file(filename):
    safe_filename = secure_filename(filename)
    if filename != safe_filename:
        # Попытка доступа с опасным именем
        abort(400, description="Invalid filename")

    filepath = os.path.join(app.config['PROCESSED_FOLDER'], safe_filename)
    if not os.path.isfile(filepath):
        abort(404, description="File not found")

    return send_from_directory(app.config['PROCESSED_FOLDER'], safe_filename)

# Запуск Flask
if __name__ == '__main__':
    app.run(debug=True)