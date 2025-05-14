import cv2
import numpy as np
import os
from PIL import Image

# Define paths
dataset_path = "Dataset"
model_path = "trainer.yml"

def get_images_and_labels():
    """Load images from Dataset folder and assign labels."""
    face_samples = []
    labels = []
    name_to_id = {}
    current_id = 0

    # Iterate through each person's folder in Dataset
    for name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, name)

        if os.path.isdir(person_folder):  # Ensure it's a folder
            if name not in name_to_id:
                name_to_id[name] = current_id
                current_id += 1

            label = name_to_id[name]

            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                
                # Convert image to grayscale
                img = Image.open(image_path).convert('L')
                img_numpy = np.array(img, 'uint8')

                face_samples.append(img_numpy)
                labels.append(label)

    return face_samples, np.array(labels), name_to_id

# Train the face recognizer
print("Training the model...")

faces, ids, name_map = get_images_and_labels()
if len(faces) == 0:
    print("No images found! Please run TakeImages.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, ids)

# Save trained model
recognizer.save(model_path)

# Save name-to-ID mapping for recognition
with open("name_labels.txt", "w") as f:
    for name, label in name_map.items():
        f.write(f"{label},{name}\n")

print(f"Model trained successfully and saved as {model_path}")
 