from flask import Flask, render_template, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64
import io
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300
labels = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"K",10:"L",11:"M",12:"N",13:"O",14:"P",15:"Q",16:"R",17:"S",18:"T",19:"U",20:"V",21:"W",22:"X",23:"Y",24:"Z",25:"J"}

# This route handles the main page
@app.route('/')
def index():
    return render_template('front.html')

# This route handles the sign language recognition endpoint
@app.route('/recognize', methods=['POST'])
def recognize_sign_language():
    video_data = request.form['video_data']
    img = decode_image(video_data)
    recognized_sign = ""

    hands, _ = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        recognized_sign = labels.get(index, 'Unknown')

    return jsonify({'recognized_sign': recognized_sign})

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    images = request.files.getlist('images')
    true_labels = request.form.getlist('labels')
    predictions = []

    for image, true_label in zip(images, true_labels):
        img = Image.open(image.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        hands, _ = detector.findHands(img)
        recognized_sign = "Unknown"

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            recognized_sign = labels.get(index, 'Unknown')

        predictions.append(recognized_sign)

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    return jsonify({'accuracy': accuracy, 'report': report})

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    app.run(debug=True)
