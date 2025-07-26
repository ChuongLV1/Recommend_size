# Sử dụng Python 3.10 slim (rất gần với môi trường của bạn)
FROM python:3.10.8-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /predict_catboostmodel

# Copy toàn bộ file từ máy local vào thư mục /app trong container
COPY . .

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# # Mở cổng 5000
# EXPOSE 5000

# Chạy app bằng gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "predict_catboostmodel:app"]

