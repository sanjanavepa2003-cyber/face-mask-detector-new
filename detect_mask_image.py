import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from google.colab.patches import cv2_imshow
from google.colab import files

# Load model
model = load_model('/content/mask_detector.h5')
print("Model loaded successfully!")

# Download face detector files
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "/content/deploy.prototxt"
)
urllib.request.urlretrieve(
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "/content/face_detector.caffemodel"
)
print("Face detector ready!")

# Upload your photo
print("Click Choose Files and upload a photo of your face...")
uploaded = files.upload()
image_path = '/content/' + list(uploaded.keys())[0]

# Detect
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

faceNet = cv2.dnn.readNet('/content/deploy.prototxt', '/content/face_detector.caffemodel')
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
faceNet.setInput(blob)
detections = faceNet.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.5:
        continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    face = image[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    (mask, withoutMask) = model.predict(face, verbose=0)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)
    label_text = f"{label}: {max(mask, withoutMask)*100:.1f}%"
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.putText(image, label_text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(f"Result: {label} — {max(mask, withoutMask)*100:.1f}% confidence")

cv2_imshow(image)
cv2.imwrite('/content/result.jpg', image)
print("Done! Right click result.jpg in files panel and download it!")