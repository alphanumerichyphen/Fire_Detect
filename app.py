from flask import Flask, render_template, request, Response
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

model = keras.models.load_model('my_model')
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            im = Image.fromarray(frame, 'RGB')
            im = im.resize((224, 224))
            img_array = image.img_to_array(im)
            img_array = np.expand_dims(img_array, axis=0) / 255
            probabilities = model.predict(img_array)[0]
            prediction = np.argmax(probabilities)

            if prediction == 0:
                cv2.putText(frame, f'Prediction: FIRE', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 100), 2)
            else:
                cv2.putText(frame, f'Prediction: NO FIRE', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 100), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response


@app.route('/')
def hello_world():  # put application's code here
    return render_template('base.html')


@app.route('/picture', methods=["GET", "POST"])
def images():
    return render_template('picture.html')


@app.route('/uploader', methods=["GET", "POST"])
def upload_files():
    path = "static/test.jpg"
    if request.method == "POST":
        f = request.files['file']
        f.save(path)

        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255
        classes = model.predict(x)
        res = np.argmax(classes[0]), max(classes[0])
        print(res)
        return render_template('result.html', res=res)


@app.route('/stream', methods=["GET", "POST"])
def stream():
    return render_template('stream.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
