# recognizer.py

import cv2
import os
import json
import numpy as np
from datetime import datetime
import atexit
from deepface import DeepFace

# Paths
dataset_path = 'dataset'
users_json = 'registered_users.json'

# Global variables to store embeddings
faces_db = []
labels_db = []

# Load registered users metadata
if os.path.exists(users_json):
    with open(users_json, 'r') as f:
        try:
            registered_users = json.load(f)
        except json.JSONDecodeError:
            registered_users = {}
else:
    registered_users = {}

# Initialize webcam
cap = cv2.VideoCapture(0)

def release_camera():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

atexit.register(release_camera)

def register_face(name):
    user_folder = os.path.join(dataset_path, name)
    os.makedirs(user_folder, exist_ok=True)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Press 'c' to capture face ({count}/20)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            face_img = frame.copy()
            filename = os.path.join(user_folder, f"{name}_{count}.jpg")
            cv2.imwrite(filename, face_img)
            count += 1
            print(f"Captured image {count}/20")
        elif key == 27:  # ESC
            break
        if count >= 20:
            break

    registered_users[name] = {
        "registered_on": str(datetime.now()),
        "images": count
    }
    with open(users_json, 'w') as f:
        json.dump(registered_users, f)

    cv2.destroyWindow("Register Face")

def load_registered_faces():
    global faces_db, labels_db
    faces_db.clear()
    labels_db.clear()
    if not os.path.exists(dataset_path):
        return
    for user in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user)
        if os.path.isdir(user_folder):
            for img_name in os.listdir(user_folder):
                img_path = os.path.join(user_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    faces_db.append(img)
                    labels_db.append(user)

def reload_faces_after_registration():
    load_registered_faces()

# Preload faces
load_registered_faces()

def gen_frames(is_registering):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_registering():
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        try:
            results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            analyses = results if isinstance(results, list) else [results]

            for res in analyses:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                face_crop = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                verified_name = "Unknown"
                min_dist = 0.9

                try:
                    candidate_embedding = DeepFace.represent(face_rgb, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                    for db_face, label in zip(faces_db, labels_db):
                        db_face_rgb = cv2.cvtColor(db_face, cv2.COLOR_BGR2RGB)
                        db_embedding = DeepFace.represent(db_face_rgb, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                        dist = np.linalg.norm(np.array(candidate_embedding) - np.array(db_embedding))
                        if dist < min_dist:
                            min_dist = dist
                            verified_name = label
                except Exception as e:
                    print("Embedding extraction failed:", e)

                text = f"{verified_name}, {res['dominant_emotion']}"
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        except Exception as e:
            print("Face analysis error:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
