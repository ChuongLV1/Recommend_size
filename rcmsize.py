import joblib
import numpy as np
import os

OUTPUT_DIR = "./save model"

# Load model
bundle = joblib.load(os.path.join(OUTPUT_DIR, "best_size_model_extended.joblib"))
pipe = bundle["pipeline"]
le = bundle["label_encoder_size"]
gmap = bundle["gender_map"]
feature_cols = bundle["feature_cols"]
canonical_order = bundle["canonical_order"]

def shift_size_by_fit(base_size, fit_code):
    try:
        idx = canonical_order.index(base_size)
        if fit_code == 0 and idx > 0:
            return canonical_order[idx - 1]  # ôm hơn → xuống 1 size
        elif fit_code == 2 and idx < len(canonical_order) - 1:
            return canonical_order[idx + 1]  # rộng hơn → lên 1 size
        return base_size
    except:
        return base_size

def predict_size_with_fit(gender_text, height_cm, weight_kg, fit_preference, apply_fit_rule=True):
    gender_code = gmap.get(gender_text.lower(), 0)
    X_new = np.array([[gender_code, height_cm, weight_kg, fit_preference]], dtype=float)
    y_pred = pipe.predict(X_new)[0]
    base_size = le.inverse_transform([y_pred])[0]
    final_size = shift_size_by_fit(base_size, fit_preference) if apply_fit_rule else base_size
    return base_size, final_size

if __name__ == "__main__":
    try:
        print("📏 Dự đoán Size Áo (Dùng mô hình mở rộng)")
        gender = input("Giới tính (Nam/Nữ): ").strip()
        height = float(input("Chiều cao (cm): "))
        weight = float(input("Cân nặng (kg): "))
        fit = input("Phong cách (ôm / vừa / rộng): ").strip().lower()
        
        fit_map = {"ôm": 0, "vừa": 1, "rộng": 2}
        if fit not in fit_map:
            raise ValueError("Phong cách phải là: ôm / vừa / rộng")
        fit_code = fit_map[fit]

        base, final = predict_size_with_fit(gender, height, weight, fit_code, apply_fit_rule=True)
        print(f"🎯 Kết quả: Size cơ bản: {base} → Sau điều chỉnh theo phong cách: {final}")

    except Exception as e:
        print("❌ Lỗi:", e)
