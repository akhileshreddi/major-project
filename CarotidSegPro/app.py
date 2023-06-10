from flask import Flask, render_template, request, send_file
import os
from model_check import predict
import cv2
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def process_image():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        return render_template('index.html', message='No file selected')

    # Save the uploaded file
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction
        prediction = predict(file_path)

        # Save the predicted image
        predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
        cv2.imwrite(predicted_image_path, prediction)

        # Convert the predicted mask to base64 string
        prediction_base64 = get_base64_from_image(prediction)

        # Convert the uploaded image to base64 string
        image_base64 = get_base64_from_image(cv2.imread(file_path))

        # Render the output page with the prediction and uploaded image
        return render_template('output.html', prediction=prediction_base64, image=image_base64)


@app.route('/download')
def download_prediction():
    prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
    return send_file(prediction_path, as_attachment=True, attachment_filename='prediction.png')


def get_base64_from_image(image):
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
