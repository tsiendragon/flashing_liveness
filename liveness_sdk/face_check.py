import numpy as np
import time


class FaceChecker:
    def __init__(self, face_check_config) -> None:
        self.bbox_thresh = face_check_config["bbox_thresh"]
        self.bbox_gap = np.array(face_check_config["bbox_gap"])
        self.bbox_reset_gap = np.array(face_check_config["bbox_reset_gap"])
        self.face_position_epsilon = 0.1

    def check_face_position(self, face_class, bbox, require_bbox):
        if face_class == "no_face":
            return False, True, "No face detected"
        elif face_class == "multi_face":
            return False, True, "Multi face detected"
        required_h = require_bbox[3] - require_bbox[1]
        required_w = require_bbox[2] - require_bbox[0]
        bbox = np.array(bbox)
        require_bbox = np.array(require_bbox)
        signed_gap = (bbox - require_bbox) / np.array(
            [required_w, required_h, required_w, required_h]
        )
        gap = np.abs(signed_gap)
        gap = np.rint(gap * 1000) / 1000

        # logging.info(f"gap: {gap}")
        reset = np.any(gap > self.bbox_reset_gap)
        if not np.all(gap < self.bbox_gap):
            valid_message = f"Position wrong: {gap}"
            return False, reset, valid_message
        else:
            valid_message = f"Position OK: {gap}"
            return True, reset, valid_message

    def __call__(self, face_class, bbox, required_bbox, frame_index):
        face_position_valid, reset_signal, valid_message = self.check_face_position(
            face_class, bbox, required_bbox
        )

        rst = [
            {
                "face_position": face_position_valid,
            },
            {
                "face_position": valid_message,
            },
        ]
        valid_face = face_position_valid
        # logging.info(f"valid face {valid_face}")
        self.update_fps(frame_index)

        return rst, reset_signal, valid_face, self.get_average_fps()

    def update_fps(self, index):
        # logging.info(f"index in update fps {index}")
        if not hasattr(self, "fps_accumulate_count"):
            self.fps_buffer = []
            self.fps_accumulate_count = 0
            self.last_index = index - 1
            self.last_time = time.time()
        self.fps_accumulate_count += 1
        if index < 10:
            update_frequent = 10
        else:
            update_frequent = int(self.get_average_fps())

        current_time = time.time()
        # logging.info(f"current_time {current_time}")
        fps = (index + 1 - self.last_index) / (current_time - self.last_time + 1e-5)
        if len(self.fps_buffer) > 10:
            self.fps_buffer.pop(0)

        if self.fps_accumulate_count > update_frequent:  #
            self.last_time = current_time
            self.last_index = index
            self.fps_accumulate_count = 0

        self.fps_buffer.append(fps)

    def get_average_fps(self):
        return np.mean(self.fps_buffer)

    def get_average_speed(self, landmarks):
        nose = landmarks[2]
        if not hasattr(self, "last_time_speed"):
            self.last_time_speed = time.time()
            self.last_nose = nose
            return 0

        elif not hasattr(self, "last_nose"):
            self.last_time_speed = time.time()
            self.last_nose = nose
            return 0
        else:
            dis = np.linalg.norm(nose - self.last_nose)
            duration = time.time() - self.last_time_speed
            self.last_nose = nose
            self.last_time_speed = time.time()
            return dis / duration
