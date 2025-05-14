import cv2
import os
import csv

def store_name():
    name = input("Enter Name: ").strip()
    return name if name.isalpha() else None

def TakeImages():
    name = store_name()
    
    if name:
        dataset_path = "Dataset"
        person_folder = os.path.join(dataset_path, name)

        # Create Dataset folder if it doesn't exist
        os.makedirs(person_folder, exist_ok=True)

        # Save the name in a CSV file
        csv_path = os.path.join(dataset_path, 'Profile.csv')
        fieldnames = ['Name']
        with open(csv_path, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Name': name})

        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                sampleNum += 1
                cv2.imwrite(f"{person_folder}/{name}_{sampleNum}.jpg", gray[y:y + h, x:x + w])

            cv2.imshow('Capturing Face for Login', img)

            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
                break

        cam.release()
        cv2.destroyAllWindows()
        print(f"Images Saved for Name: {name}")
        print(f"Images saved location: {person_folder}/")

    else:
        print("Enter a valid name (letters only)")

TakeImages()
