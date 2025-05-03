from flask import Flask, request, jsonify
from PIL import Image
import io
import pandas as pd
from ultralytics import YOLO

# تحميل الموديل المدرب
model = YOLO("best.pt")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        # تحويل الصورة إلى RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # تنفيذ التنبؤ باستخدام YOLOv8
        results = model(image)

        # استخراج النتائج بصيغة dict
        prediction = results[0].pandas().xyxy[0].to_dict(orient="records")

        return jsonify({'predictions': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
