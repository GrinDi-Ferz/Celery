import cv2
from cv2 import dnn_superres
import numpy as np

# Глобальный объект модели, загружается единоразово
scaler = dnn_superres.DnnSuperResImpl_create()
scaler.readModel('EDSR_x2.pb')
scaler.setModel("edsr", 2)

def upscale(image: np.ndarray) -> np.ndarray:
    """
    Апскейлит изображение (numpy array) без сохранения файлов на диск.

    :param image: Входное изображение в формате numpy.ndarray (цветное)
    :return: Апскейленное изображение (numpy.ndarray)
    """
    result = scaler.upsample(image)
    return result

# использования без файлов:
def example():
    # Читаем изображение разово из файла для демонстрации
    image = cv2.imread('lama_300px.png')
    if image is None:
        raise RuntimeError("Failed to read input image")

    upscaled_image = upscale(image)

    # Сохраняем результат для проверки
    cv2.imwrite('lama_600px.png', upscaled_image)


if __name__ == '__main__':
    example()