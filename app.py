from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['PNG', 'JPG', 'JPEG'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

model = load_model('ResNet50V2_model.h5',compile=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    return jsonify({
        "status":{
            "code":200,
            "message":"Success fetching the API"
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        image = request.files['image']
        if image and secure_filename(image.filename):
            #
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            image = Image.open(image_path).convert('RGB')
            image = image.resize((224,224))
            image = np.array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            result = model.predict(image)
            emotion_class = np.argmax(result)

            # Map class index to emotion label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            predicted_emotion = emotion_labels[emotion_class]

            return jsonify({
                "status":{
                    "code":200,
                    "message":"Success predicting the image"
                },
                "data": {
                    "emotion": predicted_emotion,
                    "image_path": image_path
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status":{
                "code":405,
                "message":"Bad request method"
            },
            "data": None
        }), 405
    
if __name__ == "__main__":
    app.run()