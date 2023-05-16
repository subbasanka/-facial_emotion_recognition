from flask import Flask, render_template, request, Response
import cv2
from keras.models import load_model
import numpy as np
from camera import VideoCamera

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

video = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')

# def index():
#     return render_template('index.html')

def gen(video):
    while True:
        # success, image = video.read()
        frame = video.get_frame()
        # ret, jpeg = cv2.imencode('.jpg', image)
        # frame = video.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']

    img.save('static/file.jpg')

    ####################################
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', img1)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)

    except:
        pass

    #####################################

    try:
        image = cv2.imread('static/cropped.jpg', 0)
    except:
        image = cv2.imread('static/file.jpg', 0)

    image = cv2.resize(image, (48,48))

    image = image/255.0

    image = np.reshape(image, (1,48,48,1))

    model = load_model('model.h5')

    prediction = model.predict(image)

    label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('after.html', data=final_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


