import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime

# Load encodings
with open("encodings.pkl", "rb") as f:
    encodeListKnown, names = pickle.load(f)

def markAttendance(name):
    with open("attendance.csv", "a+") as f:
        f.seek(0)
        data = f.readlines()
        nameList = [line.split(',')[0] for line in data]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f"\n{name},{dtString}")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

