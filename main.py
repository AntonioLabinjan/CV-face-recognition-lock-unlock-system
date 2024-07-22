import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from transformers import CLIPProcessor, CLIPModel
import torch

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Function to add a known face
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))  # Normalize the embedding
    known_face_names.append(name)

# Adding known faces (example)
add_known_face("antonio.jpg", "Antonio")
add_known_face("antonio1.jpg", "Antonio")
add_known_face("antonio2.jpg", "Antonio")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

lock_status = "Locked"
is_unlocked = False  # Flag to track if the system is unlocked
camera_active = True  # Flag to control camera activity

def generate_frames():
    global lock_status, is_unlocked, camera_active
    while True:
        if not camera_active:
            continue  # Skip the loop if the camera is turned off
        
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if not is_unlocked:
            lock_status = "Locked"  # Default status

            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                inputs = processor(images=face_image_rgb, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                face_embedding /= np.linalg.norm(face_embedding)  # Normalize the embedding

                # Find the best match for the detected face
                distances = np.linalg.norm(known_face_encodings - face_embedding, axis=1)
                min_distance_index = np.argmin(distances)
                name = "Unknown"
                if distances[min_distance_index] < 0.6:  # Adjusted threshold
                    name = known_face_names[min_distance_index]
                    lock_status = "Unlocked"  # Unlock if face is recognized
                    is_unlocked = True  # Set flag to indicate unlocked state

                # Draw a rectangle around the face and label it
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lock_status')
def lock_status_endpoint():
    return jsonify({'status': lock_status})

@app.route('/lock', methods=['POST'])
def lock():
    global lock_status, is_unlocked
    lock_status = "Locked"
    is_unlocked = False
    return jsonify({'status': lock_status})

@app.route('/unlock', methods=['POST'])
def unlock():
    global lock_status, is_unlocked
    lock_status = "Unlocked"
    is_unlocked = True
    return jsonify({'status': lock_status})

@app.route('/camera_on', methods=['POST'])
def camera_on():
    global camera_active
    camera_active = True
    return jsonify({'camera_active': camera_active})

@app.route('/camera_off', methods=['POST'])
def camera_off():
    global camera_active
    camera_active = False
    return jsonify({'camera_active': camera_active})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
