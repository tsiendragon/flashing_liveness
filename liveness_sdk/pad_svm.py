import  cv2
import numpy as np
from skimage import feature
import joblib
from sklearn.svm import LinearSVC
import os

class svm_PresentationAttackDetection:
    """This class is used to detect presentation attack in the image."""

    def __init__(self) -> None:
        pass

    def __call__(self, image_dict: dict) -> [bool,bool]:
        color=["blue","green","red"]
        resize_size = 128
        hog_features = []
        hog_image = []

        win_size = (resize_size, resize_size)  # Window size for HOG descriptor
        block_size = (8, 8)  # Block size for HOG descriptor
        block_stride = (4, 4)  # Block stride for HOG descriptor
        cell_size = (4, 4)  # Cell size for HOG descriptor
        nbins = 9  # Number of bins for HOG descriptor

        image=image_dict

        for i in image:
            image[i]=cv2.resize(image[i],(resize_size,resize_size))
            if i!="white":
                image[i]=image[i]-image["white"]
                new_img=cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
                new_img=new_img/255
                new_img=np.clip(new_img, 0, 255)
                hog_fea, hog_img = feature.hog(new_img, orientations=nbins, pixels_per_cell=cell_size,
                                                  cells_per_block=block_size, block_norm='L2-Hys', visualize=True)
                hog_features.append(hog_fea)

        hog_features=np.array(hog_features)

        if len(hog_features)<3:
            return [True,True]

        else:
            clf = LinearSVC()
        # 下面的代码是保存模型的
        model_path="./"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        clf = joblib.load(model_path+'model')
        result = clf.predict(hog_features)
        res=np.sum(result)
        if res>=2:
            return [False,False]
        else:
            return [True,False]