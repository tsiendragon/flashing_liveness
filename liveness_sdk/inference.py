from typing import Any
import cv2
import os
import numpy as np

package_dir = os.path.dirname(os.path.abspath(__file__))


class FaceDetect:
    protoPath = os.path.join("./face_detector", "deploy.prototxt")
    protoPath = os.path.join(package_dir, protoPath)
    modelPath = os.path.join(
        "./face_detector", "res10_300x300_ssd_iter_140000.caffemodel"
    )
    modelPath = os.path.join(package_dir, modelPath)

    def __init__(self, face_confidence_threshold=0.5) -> None:
        self.net = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        self.face_confidence_threshold = face_confidence_threshold

    def __call__(self, image) -> Any:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        (h, w) = image.shape[:2]
        kept_box = detections[0, 0, :, 2] > self.face_confidence_threshold
        detections = detections[:, :, kept_box, :]
        if detections.shape[2] > 1:
            face_class = "multi_face"
        elif detections.shape[2] == 0:
            face_class = "no_face"
        else:
            face_class = "single_face"
        box = np.array([0, 0, 0, 0])
        if face_class == "single_face":
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        return face_class, box.astype("int")
