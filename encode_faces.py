import face_recognition
import os
import pickle

path = "images"
images = []
names = []

for file in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{file}")
    images.append(img)
    names.append(os.path.splitext(file)[0])

encodeList = []

for img in images:
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)

with open("encodings.pkl", "wb") as f:
    pickle.dump((encodeList, names), f)

print("Encoding Complete!")
