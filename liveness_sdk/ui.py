# import logging
from typing import List

import cv2
import numpy as np

color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
}


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    color = mc.to_rgb(color)
    return color


class LivenessSdkUi:
    def __init__(self, frame, config) -> None:

        frame_h, frame_w = frame.shape[:2]

        self.config = config
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.center_x = int(frame_w / 2)
        self.center_y = int(frame_h / 2)
        self.get_target_face_position(frame, face_box_config=config["face_box"])
        self.face_position_message_loc = (0, 20)
        self.fps_loc = (0, 45)
        self.face_position_epsilon = 0.04

        self.locations = {
            "index": (0.14, 0.06),
            "fps": (0, 0.06),
            "prompt": (0.25, 0.9),
            "speed": (0.7, 0.06),
        }
        self.colors = {  # BGR
            "OK": (0, 255, 0),
            "face_position_error": (0, 0, 255),
            "fps": (255, 0, 255),
            "index": (255, 0, 255),
            "prompt": (255, 0, 255),
            "speed": (255, 0, 255),
        }
        # from ratio to pixel
        for k, v in self.locations.items():
            v = (int(v[0] * self.frame_w), int(v[1] * self.frame_h))
            self.locations[k] = v

    def get_target_face_position(self, frame, face_box_config):
        img_h, img_w = frame.shape[:2]
        center_x = int(img_w * 0.5)
        center_y = int(img_h * 0.5)
        face_ratio = face_box_config["face_ratio"]
        axis_y = int(img_h * face_box_config["height"])
        axis_x = int(axis_y / face_ratio)
        required_bbox = [
            center_x - axis_x,
            center_y - axis_y,
            center_x + axis_x,
            center_y + axis_y,
        ]
        self.required_bbox = required_bbox
        self.axis_x = axis_x
        self.axis_y = axis_y

    def update_face_status(
        self,
        frame,
        face_check_rst: List[dict],
        prompt_message: str = None,
        background_color="white",
        fps=None,
    ):
        result = face_check_rst[0]
        message = face_check_rst[1]
        face_position_valid = result["face_position"]

        assert frame.shape[:2] == (
            self.frame_h,
            self.frame_w,
        ), f"image shape changed, expect {(self.frame_h, self.frame_w)}, but got {frame.shape[:2]}"
        if face_position_valid:
            # logging.info("is OK")
            color = self.colors["OK"]
        elif not face_position_valid:
            color = self.colors["face_position_error"]
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        mask = np.zeros((self.frame_h*3,self.frame_w*5,3), dtype=np.uint8)

        half_h=self.frame_h//2
        half_w=self.frame_w//2

        mask[self.frame_h:2*self.frame_h,2*self.frame_w
:3*self.frame_w]=frame

        mask_h,mask_w=mask.shape[:2]
        self.center_x = int(mask_w / 2)
        self.center_y = int(mask_h / 2)

        frame=mask


        frame = cv2.ellipse(
            frame,
            (self.center_x, self.center_y),
            (self.axis_x, self.axis_y),
            0.0,
            0.0,
            360.0,
            color,
            4,
        )
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        ellipse_mask = cv2.ellipse(
            mask,
            (self.center_x, self.center_y),
            (self.axis_x, self.axis_y),
            0.0,
            0.0,
            360.0,
            255,
            -1,
        )
        frame[ellipse_mask != 255] = color_mapping[background_color]

        # frame = self.update_face_position_guide(frame, result)
        if fps is not None:
            frame = self.write_fps_message(frame, fps)
        position_message = message["face_position"]
        frame = self.write_face_position_message(frame, position_message, color)

        if prompt_message is not None:
            frame = self.write_prompt_message(frame, prompt_message)
        return frame

    def write_face_position_message(self, frame, position_message, color):
        frame = cv2.putText(
            frame,
            position_message,
            self.face_position_message_loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return frame

    def write_fps_message(self, frame, fps):
        fps = round(float(fps), 2)
        message = f"fps: {fps}"
        frame = cv2.putText(
            frame,
            message,
            self.fps_loc,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["fps"],
            1,
        )
        return frame

    def write_moving_speed(self, frame, speed):
        speed = round(float(speed), 2)
        message = f"speed: {speed}"
        frame = cv2.putText(
            frame,
            message,
            self.locations["speed"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["speed"],
            1,
        )
        return frame

    def write_prompt_message(self, frame, prompt_message):
        frame = cv2.putText(
            frame,
            prompt_message,
            self.locations["prompt"],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.colors["prompt"],
            2,
        )
        return frame
