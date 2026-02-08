import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ====== Load Known Faces ======
known_faces = []
known_names = []

path = os.path.join(os.getcwd(), "known_faces")
if not os.path.exists(path):
    os.makedirs(path)
    print(f"[INFO] 'known_faces' folder created at: {path}")
    print("[INFO] Please add face images (e.g., shashank.jpg) into this folder.")

for filename in os.listdir(path): 
    image_path = os.path.join(path, filename)
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"[INFO] Loaded face for {filename}")
        else:
            print(f"⚠ No face found in {filename}, skipping.")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

if len(known_faces) == 0:
    print("❌ No known faces loaded. Please add images to 'known_faces' folder.")
    exit()

# ====== Initialize Attendance ======
attendance_file = "Attendance.xlsx"
try:
    attendance = pd.read_excel(attendance_file)
except:
    attendance = pd.DataFrame(columns=["Name", "Time"])

# ====== Open Camera ======
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

                if name not in attendance["Name"].values:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance.loc[len(attendance)] = [name, now]
                    print(f"✅ Attendance marked: {name} at {now}")

    # Display webcam
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Save Attendance ======
attendance.to_excel(attendance_file, index=False)
print(f"📁 Attendance saved to {attendance_file}")

cap.release()
cv2.destroyAllWindows()