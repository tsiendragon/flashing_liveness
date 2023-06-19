import cv2
import numpy as np
import time
from imutils.video import VideoStream
import os
from ui import LivenessSdkUi
from inference import FaceDetect
from utils import square_crop_image, create_datetime_str
from options import options
from face_check import FaceChecker
from pad import PresentationAttackDetection
from pad_svm import svm_PresentationAttackDetection

import yaml

package_dir = os.path.dirname(os.path.abspath(__file__))


class FlashingStatus:
    """change the background color with predefined colors, and collect the flashing images
    Determine when to start and stop flashing & collecting images
    """

    def __init__(
        self,
        save_dir=None,
        flashing_colors=["white", "blue", "black"],
        flashing_interval=0.2,
    ) -> None:
        self.save_dir = os.path.join(os.path.dirname(package_dir), save_dir)
        self.flashing_colors = flashing_colors
        self.flashing_interval = flashing_interval
        self.fps_hist = []
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_color = "white"
        self.current_color_index = -1
        self.current_collect = False
        self.previous_time = 0
        self.image_dict = {}
        self.image_selected = False
        self.current_message = "Capturing Images: Keep still"
        self.status = "In_progress"
        self.capture_face=False


    def update(self, frame: np.ndarray, fps=0, stage: int = 0):
        # stage 0: prepare
        # stage 1: flashing
        # stage 2: save and stop
        # if passed prepare stage
        # changing the light
        # avrage fps
        #print("imge dict", self.image_dict.keys())
        if stage == 0:
            self.fps_hist.append(fps)
        if stage == 1:
            fps = np.mean(self.fps_hist)
            assert (
                fps > 0.5 / self.flashing_interval
            ), f"current fps did not support the flashing interval {self.flashing_interval}"
            if (
                time.time() - self.previous_time > self.flashing_interval and self.image_selected==True
            ):  # change color
                self.current_color_index += 1
                if self.current_color_index >= len(self.flashing_colors):
                    self.status = "Done"
                    self.current_message = "Capturing Images: Finished"
                else:
                    self.current_color = self.flashing_colors[self.current_color_index]
                    self.previous_time = time.time()
                    self.image_selected = False
            if time.time() - self.previous_time > self.flashing_interval / 2:
                if self.image_selected is False:  # if already take 1, skip it

                    #save the whole picture
                    if not self.capture_face:
                        self.image_dict[self.current_color] = frame

                    else:
                    #save the face
                        (h, w) = frame.shape[:2]
                        numclass, facebox = face_detector(frame)
                        (startX, startY, endX, endY) = facebox
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        face = frame[startY:endY, startX:endX]
                        self.image_dict[self.current_color] = face

                    self.image_selected = True

    def save(self) -> None:
        sub_folder = create_datetime_str()
        dst = os.path.join(self.save_dir, sub_folder)
        os.makedirs(dst, exist_ok=True)
        for k in self.image_dict:
            cv2.imwrite(os.path.join(dst, f"{k}.png"), self.image_dict[k])


# load the face detector
config = os.path.join(package_dir, options.config)

with open(config, "r") as stream:
    config = yaml.safe_load(stream)


face_detector = FaceDetect(
    face_confidence_threshold=config["face_detect"]["face_confidence_threshold"]
)
face_check = FaceChecker(config["face_check"])
flashing_status = FlashingStatus(
    save_dir=config["data"]["save_dir"], **config["flashing"]
)
pad = PresentationAttackDetection()
video_stream = VideoStream(src=0).start()
print("[INFO] loading face detector...")


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


whether_break = False


def main():
    # steps, first guide user to put face into the circle
    # second, prompt user keeping still
    # third, changing background color with predefined colors
    # fourth, save the images for each color
    # fifth, gather all the images collected and send to the model
    # sixth, analysis the result and send the result to user
    count = 0
    stage = 0
    while True:
        count += 1
        frame = video_stream.read()
        original_frame = frame.copy()
        frame = square_crop_image(frame)
        frame = cv2.flip(frame, 1)



        (h, w) = frame.shape[:2]
        #print(frame.shape)


        base_ui = LivenessSdkUi(frame, config["UI"])

        # face detect
        face_class, bbox = face_detector(frame)
        face_check_rst, reset_signal, valid_face, fps = face_check(
            face_class, bbox, base_ui.required_bbox, count
        )
        if valid_face:
            stage = 1
        if reset_signal:
            stage = 0

        flashing_status.update(original_frame, fps, stage=stage)
        background_color = (
            flashing_status.current_color
            if flashing_status.status == "In_progress"
            else "white"
        )



        frame = base_ui.update_face_status(
            frame,
            face_check_rst,
            fps=fps,
            background_color=background_color,
            prompt_message=flashing_status.current_message,
        )

        cv2.imshow("Face Capture", frame)

        # 检测键盘按键，按下 "q" 键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            whether_break = True
            break

        if cv2.waitKey(1) & 0xFF == ord("r"):
            print("[INFO] new round")
            break
        #print("flashing_status.status", flashing_status.status)
        if flashing_status.status == "Done":
            video_stream.stop()
            break

    flashing_status.save()
    is_attack, re_captrue = pad(flashing_status.image_dict)



    if re_captrue == True:
        message = "Please let your background be darker"
    else:
        message = f"Attack: {is_attack}"

    frame = square_crop_image(original_frame)

    frame = base_ui.update_face_status(
        frame,
        face_check_rst,
        fps=fps,
        prompt_message=message,
    )
    cv2.destroyAllWindows()
    cv2.namedWindow("Face Capture")

    cv2.imshow("Face Capture", frame)
    cv2.waitKey(0)  # Wait for a key press (optional)

    input("press any key to exit")


if __name__ == "__main__":
    main()
# do a bit of cleanup

cv2.destroyAllWindows()
