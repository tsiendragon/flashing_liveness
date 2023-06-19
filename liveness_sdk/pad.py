import cv2
import numpy as np
import dlib


class PresentationAttackDetection:
    """This class is used to detect presentation attack in the image."""

    def __init__(self) -> None:
        pass


    def test_point(self,cur_color,first_color,second_color,landmarks,x,y):

        nose_x = landmarks.part(x).x
        nose_y = landmarks.part(y).y

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



    def test_face_boundary(self,cur_color,first_color,second_color,landmarks):
        left_x = landmarks.part(0).x
        left_y = landmarks.part(0).y

        # You can use the nose landmark as a reference to define the region
        left_region_x = left_x - 10  # Example: Subtracting 10 pixels from the nose x-coordinate
        left_region_y = left_y - 10  # Example: Subtracting 10 pixels from the nose y-coordinate
        left_region_width = 20  # Example: Setting the region width to 20 pixels
        left_region_height = 20  # Example: Setting the region height to 20 pixels

        left_channel_1= cur_color[left_region_y:left_region_y + left_region_height,
                      left_region_x:left_region_x + left_region_width]
        left_channel_2= first_color[left_region_y:left_region_y + left_region_height,
                      left_region_x:left_region_x + left_region_width]
        left_channel_3= second_color[left_region_y:left_region_y + left_region_height,
                      left_region_x:left_region_x + left_region_width]



        right_x = landmarks.part(16).x
        right_y = landmarks.part(16).y

        # You can use the nose landmark as a reference to define the region
        right_region_x = right_x - 10  # Example: Subtracting 10 pixels from the nose x-coordinate
        right_region_y = right_y - 10  # Example: Subtracting 10 pixels from the nose y-coordinate
        right_region_width = 20  # Example: Setting the region width to 20 pixels
        right_region_height = 20  # Example: Setting the region height to 20 pixels

        right_channel_1= cur_color[right_region_y:right_region_y + right_region_height,
                      right_region_x:right_region_x + right_region_width]
        right_channel_2= first_color[right_region_y:right_region_y + right_region_height,
                      right_region_x:right_region_x + right_region_width]
        right_channel_3= second_color[right_region_y:right_region_y + right_region_height,
                      right_region_x:right_region_x + right_region_width]


        if np.mean(left_channel_1)<np.mean(right_channel_1):
            if np.mean(left_channel_2)<np.mean(right_channel_2) and np.mean(left_channel_3)<np.mean(right_channel_3):
                return True
            else:
                return False
        else:
            if np.mean(left_channel_2)>np.mean(right_channel_2) and np.mean(left_channel_3)>np.mean(right_channel_3):
                return True
            else:
                return False

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

        b2,g2,r2=b-b2,g-g2,r-r2
        b3,g3,r3=b-b3,g-g3,r-r3
        b4,g4,r4=b-b4,g-g4,r-r4

        b2,g2,r2=np.clip(b2,0,255),np.clip(g2,0,255),np.clip(r2,0,255)
        b3,g3,r3=np.clip(b3,0,255),np.clip(g3,0,255),np.clip(r3,0,255)
        b4,g4,r4=np.clip(b4,0,255),np.clip(g4,0,255),np.clip(r4,0,255)

        # Assuming you want to define a region around the nose







        count=0


        if self.test_point(b2,g2,r2,landmarks,41,33) and self.test_point(b2,g2,r2,landmarks,45,33) and self.test_point(b2,g2,r2,landmarks,27,27):
            count+=1

        if self.test_point(g3,b3,r3,landmarks,41,33) and self.test_point(g3,b3,r3,landmarks,45,33) and self.test_point(g3,b3,r3,landmarks,27,27):
            count+=1

        if self.test_point(r4,b4,g4,landmarks,41,33) and self.test_point(r4,b4,g4,landmarks,45,33) and self.test_point(r4,b4,g4,landmarks,27,27):

            count+=1

        if count ==3 :
            return [False,False]
        elif count<=1:
            return [True,False]
        else:
            return [True, True]


