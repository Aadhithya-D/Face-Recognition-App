import cv2
import numpy as np
import face_recognition

cap = cv2.VideoCapture(0)
imgAadhi = face_recognition.load_image_file("Images/aadhi.jpg")
imgAadhi = cv2.cvtColor(imgAadhi, cv2.COLOR_BGR2RGB)
faceEncoding = face_recognition.face_encodings(imgAadhi)[0]

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    result = face_recognition.compare_faces(face_encodings, faceEncoding)
    if result == [True]:
        cv2.putText(frame, "Aadhithya", (left, bottom), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
