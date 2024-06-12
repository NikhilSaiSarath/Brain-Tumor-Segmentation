from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

model = load_model('unet_brain_mri_seg.hdf5', compile=False)

im_height = 256
im_width = 256

def preprocess_input(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0
    return img_array

def predict_mask(input_image):
    return model.predict(input_image)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img_np = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = image.array_to_img(img_np, data_format=None, scale=True)
        input_image = preprocess_input(img)
        predicted_mask = predict_mask(input_image)

        plt.figure(figsize=(12, 12))

        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(img_np))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(predicted_mask) > 0.5)
        plt.title('Prediction')

        output_bytes = BytesIO()
        plt.savefig(output_bytes, format='png')
        output_bytes.seek(0)

        plt.close()

        return send_file(output_bytes, mimetype='image/png', as_attachment=False, download_name='output.png')

    return render_template('index.html', output=None)

if __name__ == '__main__':
    app.run(debug=True)
