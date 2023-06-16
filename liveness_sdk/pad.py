import cv2
import numpy as np

class PresentationAttackDetection:
    """This class is used to detect presentation attack in the image."""

    def __init__(self) -> None:
        pass

    def __call__(self, image_dict: dict) -> [bool,bool]:
        color=["blue","green","red"]

        #if we did not capture the enough faces, restart the program
        if len(image_dict)<4:
            return [True,True]

        h,w=224,224
        h,w=h//3,w//3
        #resize the img
        for k in image_dict:
            image_dict[k]=cv2.resize(image_dict[k], (224, 224))
            image_dict[k]=image_dict[k].astype(np.float32)
            image_dict[k]=image_dict[k][h:2*h,w:2*w]

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


        count=0

        if np.mean(g2+r2)<0.5:
            count+=1

        if np.mean(b3+r3)<0.5:
            count+=1

        if np.mean(b4+g4)<0.5:
            count+=1

        if count == 3:
            return [False,False]
        elif count==0:
            return [True,False]
        else:
            return [True,True]

