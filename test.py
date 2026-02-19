import io
import pytest
from task_serves import app, upscale_task  # поправьте импорт под ваш проект
from celery.result import AsyncResult

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Для тестов с Celery лучше запускать задачу синхронно, чтобы получить результат сразу
@pytest.fixture(autouse=True)
def celery_eager(monkeypatch):
    # Это включает выполнение задач Celery "сразу" в тестах, без брокера
    monkeypatch.setattr(upscale_task, 'apply_async', lambda *args, **kwargs: upscale_task(*args[0]))

def test_upload_file_success(client):
    # Создаем простой тестовый jpg в памяти (1x1 px)
    import cv2
    import numpy as np

    img = np.zeros((1,1,3), dtype=np.uint8)
    success, encoded = cv2.imencode('.jpg', img)
    assert success

    data = {
        'file': (io.BytesIO(encoded.tobytes()), 'test.jpg')
    }
    response = client.post('/upscale', data=data, content_type='multipart/form-data')
    assert response.status_code == 202
    json_data = response.get_json()
    assert 'task_id' in json_data

def test_upload_file_no_file(client):
    response = client.post('/upscale', data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == 'No file uploaded'

def test_upload_file_empty_filename(client):
    data = {
        'file': (io.BytesIO(b'some bytes'), '')
    }
    response = client.post('/upscale', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == 'Empty filename'

def test_get_task_status_success(client):
    # Генерируем простой черный картинку в байтах и запускаем задачу напрямую
    import cv2
    import numpy as np

    img = np.zeros((1,1,3), dtype=np.uint8)
    success, encoded = cv2.imencode('.jpg', img)
    assert success
    image_bytes = encoded.tobytes()

    # Запускаем задачу и сразу получаем id
    task = upscale_task.apply_async(args=[image_bytes])
    task_id = task.id if hasattr(task, 'id') else 'test_task'

    # Включаем eager режим: в реальном тесте вы можете просто вызвать функцию напрямую
    # Теперь делаем get запрос к статусу
    response = client.get(f'/tasks/{task_id}')
    assert response.status_code == 200

    if 'image/jpeg' in response.content_type:
        # Получили картинку
        assert response.data.startswith(b'\xff\xd8')  # jpeg header
    else:
        # Получаем статус
        json_data = response.get_json()
        assert json_data['task_id'] == task_id
        assert 'status' in json_data

def test_get_task_status_invalid(client):
    response = client.get('/tasks/invalid_task_id')
    # Вернется статус (PENDING, FAILURE или тп)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'status' in json_data