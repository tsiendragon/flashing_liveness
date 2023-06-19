import cv2
import numpy as np
import dlib


class PresentationAttackDetection:
    """This class is used to detect presentation attack in the image."""

    def __init__(self) -> None:
        pass


    def test_nose(self,cur_color,first_color,second_color,landmarks):

        nose_x = landmarks.part(30).x
        nose_y = landmarks.part(30).y

        # You can use the nose landmark as a reference to define the region
        nose_region_x = nose_x - 10  # Example: Subtracting 10 pixels from the nose x-coordinate
        nose_region_y = nose_y - 10  # Example: Subtracting 10 pixels from the nose y-coordinate
        nose_region_width = 20  # Example: Setting the region width to 20 pixels
        nose_region_height = 20  # Example: Setting the region height to 20 pixels

        channel_1= cur_color[nose_region_y:nose_region_y + nose_region_height,
                      nose_region_x:nose_region_x + nose_region_width]
        channel_2= first_color[nose_region_y:nose_region_y + nose_region_height,
                      nose_region_x:nose_region_x + nose_region_width]
        channel_3= second_color[nose_region_y:nose_region_y + nose_region_height,
                      nose_region_x:nose_region_x + nose_region_width]

        if np.mean(channel_1)<np.mean(channel_2) and np.mean(channel_1)<np.mean(channel_3):
            return True



    def test_eye(self,cur_color,first_color,second_color,landmarks):


        # You can use the left eye landmark as a reference to define the region
        begin_x = landmarks.part(37).x
        begin_y = landmarks.part(37).y
        end_x = landmarks.part(40).x
        end_y = landmarks.part(40).y

        channel_1 = cur_color[begin_y:end_y, begin_x:end_x]
        channel_2 = first_color[begin_y:end_y, begin_x:end_x]
        channel_3 = second_color[begin_y:end_y, begin_x:end_x]

        if np.mean(channel_1)<np.mean(channel_2) and np.mean(channel_1)<np.mean(channel_3):
            return True

        # You can use the right eye landmark as a reference to define the region
        begin_x = landmarks.part(43).x
        begin_y = landmarks.part(43).y
        end_x = landmarks.part(46).x
        end_y = landmarks.part(46).y

        channel_1 = cur_color[begin_y:end_y, begin_x:end_x]
        channel_2 = first_color[begin_y:end_y, begin_x:end_x]
        channel_3 = second_color[begin_y:end_y, begin_x:end_x]

        if np.mean(channel_1)<np.mean(channel_2) and np.mean(channel_1)<np.mean(channel_3):
            return True




    def __call__(self, image_dict: dict) -> [bool,bool]:
        color=["blue","green","red"]

        #if we did not capture the enough faces, restart the program
        if len(image_dict)<4:
            return [True,True]

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./face_detector/shape_predictor_68_face_landmarks.dat")

        image = image_dict["white"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        face = detector(gray, 0)
        if len(face)==0:
            return [False,True]
        landmarks = predictor(image, face[0])
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        #resize the img
        for k in image_dict:
            image_dict[k]=image_dict[k].astype(np.float32)


        # Split the image into its color channels
        b, g, r = cv2.split(image_dict["white"])
        b2, g2, r2 = cv2.split(image_dict["blue"])
        b3, g3, r3 = cv2.split(image_dict["green"])
        b4, g4, r4 = cv2.split(image_dict["red"])

        b2,g2,r2=b2-b,g2-g,r2-r
        b3,g3,r3=b3-b,g3-g,r3-r
        b4,g4,r4=b4-b,g4-g,r4-r

        b2,g2,r2=np.clip(b2,0,255),np.clip(g2,0,255),np.clip(r2,0,255)
        b3,g3,r3=np.clip(b3,0,255),np.clip(g3,0,255),np.clip(r3,0,255)
        b4,g4,r4=np.clip(b4,0,255),np.clip(g4,0,255),np.clip(r4,0,255)

        # Assuming you want to define a region around the nose







        count=0


        if self.test_nose(b2,g2,r2,landmarks) or self.test_eye(b2,g2,r2,landmarks):
            count+=1

        if self.test_nose(g3,b3,r3,landmarks) or self.test_eye(g3,b3,r3,landmarks):
            count+=1

        if self.test_nose(r4,b4,g4,landmarks) or self.test_eye(r4,b4,g4,landmarks):
            count+=1

        if count >=2 :
            return [False,False]
        else:
            return [True,False]

