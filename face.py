import cv2
import numpy as np
import os
import requests

def update_data_to_blynk(value):
    iot_url = f"https://blynk.cloud/external/api/update?token=NqiHAp5LutuwYnFSN7hNVPZ4cAje5_NQ&V0={value}"
    try:
        response = requests.get(iot_url)
        if response.status_code == 200:
            print(f"Data sent to Blynk: {value}")
        else:
            print(f"Failed to send data to Blynk. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error while sending data to Blynk: {e}")

# Load trained model
model_path = "trainer.yml"
name_labels_path = "name_labels.txt"
face_cascade_path = "haarcascade_frontalface_default.xml"

# Check if model exists
if not os.path.exists(model_path):
    print("Error: Trained model not found! Please run train.py first.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load trained LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load name-label mappings
name_map = {}
if os.path.exists(name_labels_path):
    with open(name_labels_path, "r") as f:
        for line in f:
            label, name = line.strip().split(',')
            name_map[int(label)] = name

# Start video capture
cam = cv2.VideoCapture(0)

print("Press 'SPACE' to capture and recognize. Press 'q' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Camera not found!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition - Press SPACE", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Capture image on SPACE press
    if key == 32:  # ASCII code for SPACE
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # Recognize face
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 70:  # Adjust confidence threshold if needed
                name = name_map.get(label, "Unknown")
            else:
                name = "Unknown"

            print(f"Captured! Recognized: {name} (Confidence: {confidence:.2f})")

            # Display name on frame
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            update_data_to_blynk(name)

            # Show the captured frame for a moment
            cv2.imshow("Captured Face", frame)
            cv2.waitKey(1000)  # Show for 1 second

    # Exit on 'q' key press
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
