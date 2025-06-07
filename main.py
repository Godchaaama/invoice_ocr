import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
from best_model import optimal_model
from output import export

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không có file được chọn')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file')
            return redirect(request.url)
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Đọc file ảnh trực tiếp từ bộ nhớ mà không lưu xuống đĩa
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    flash('Lỗi: File ảnh không hợp lệ')
                    return redirect(request.url)
                result, none_counter = optimal_model(image)
                export(result)
                return render_template('result.html', result=result, none_counter=none_counter)
            except Exception as e:
                flash(f'Lỗi xử lý hình ảnh: {str(e)}')
                return redirect(request.url)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)