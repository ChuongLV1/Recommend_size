import pandas as pd
import joblib
import numpy as np

# Load mô hình và label encoder
pipeline = joblib.load("../Recommend_size/rf_full_pipeline.pkl")
label_encoder = joblib.load("../Recommend_size/rf_full_label_encoder.pkl")

# Giá trị trung bình của các feature còn lại theo giới tính
mean_values = {
    'nam': {
        'chest': 100,
        'waist': 90,
        'shoulder': 60
    },
    'nữ': {
        'chest': 110,
        'waist': 80,
        'shoulder': 60
    }
}

def predict_size(height, weight, gender, chest=None, waist=None, shoulder=None):
    gender = gender.strip().lower()
    
    if gender not in mean_values:
        raise ValueError("Giới tính phải là 'nam' hoặc 'nữ'.")

    # Điền giá trị thiếu bằng trung bình theo giới tính
    chest = chest if chest is not None else mean_values[gender]['chest']
    waist = waist if waist is not None else mean_values[gender]['waist']
    shoulder = shoulder if shoulder is not None else mean_values[gender]['shoulder']
    
    # Tính BMI
    BMI = weight / ((height / 100) ** 2)

    # Tạo dataframe cho input
    input_df = pd.DataFrame([{
        'height': height,
        'weight': weight,
        'chest': chest,
        'waist': waist,
        'shoulder': shoulder,
        'gender': gender,
        'BMI': BMI
    }])

    # Dự đoán
    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return pred_label

# Chạy local bằng giao diện dòng lệnh
if __name__ == "__main__":
    try:
        height = float(input("Nhập chiều cao (cm): "))
        weight = float(input("Nhập cân nặng (kg): "))
        gender = input("Nhập giới tính (Nam/Nữ): ").strip().lower()

        if gender not in ['nam', 'nữ', 'male', 'female']:
            print("⚠️ Giới tính không hợp lệ. Chỉ chấp nhận: Nam/Nữ hoặc male/female.")
        else:
            # Các input tùy chọn
            chest_input = input("Nhập chiều dài lưng (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            waist_input = input("Nhập vòng ngực (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            shoulder_input = input("Nhập ngang vai (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            # Chuyển đổi nếu có
            chest = float(chest_input) if chest_input else None
            shoulder = float(shoulder_input) if shoulder_input else None
            waist = float(waist_input) if waist_input else None

            # Dự đoán
            result = predict_size(height, weight, gender, chest, waist, shoulder)
            print(f"\n🎯 Recommended size: {result}")
    except Exception as e:
        print("❌ Lỗi khi nhập hoặc xử lý dữ liệu:", e)