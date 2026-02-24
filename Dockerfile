FROM python:3.10-slim

WORKDIR /app

# Скопировать файлы проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создать папки для файлов
RUN mkdir -p uploads processed

# Экспонируем порт Flask
EXPOSE 5000

# Команда по умолчанию запускает Flask (в production лучше использовать gunicorn)
CMD ["flask", "run", "--host=0.0.0.0"]