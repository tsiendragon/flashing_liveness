import  cv2
import numpy as np
from skimage import feature, exposure
import os
import joblib
from sklearn.svm import LinearSVC

color=["white","blue","green","red"]


path=[]
path.append("./data/real")
path.append("./data/fake")




resize_size=128
hog_features = []
hog_image=[]
train_data=[]
label=[]


win_size = (resize_size, resize_size)  # Window size for HOG descriptor
block_size = (8, 8)  # Block size for HOG descriptor
block_stride = (4, 4)  # Block stride for HOG descriptor
cell_size = (4, 4)  # Cell size for HOG descriptor
nbins = 9  # Number of bins for HOG descriptor



for j in range(2):
    for k in range(12):
        folder=str(k)
        image = []
        gray_image = []
        for i in range(4):
            image.append(cv2.imread(os.path.join(path[j],folder,"{}.png".format(color[i]))))
            image[i]=cv2.resize(image[i],(resize_size,resize_size))
            if i>0:
                image[i]=image[i]-image[0]



                new_img=cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
                new_img=new_img/255
                new_img=np.clip(new_img, 0, 255)
                gray_image.append(new_img)
                hog_fea, hog_img = feature.hog(gray_image[i-1], orientations=nbins, pixels_per_cell=cell_size,
                                                  cells_per_block=block_size, block_norm='L2-Hys', visualize=True)
                train_data.append(hog_fea)
                if j==0:
                    label.append(1)
                else:
                    label.append(0)

    #cv2.imshow("HOG Image", hog_image[2])

train_data=np.array(train_data)
label=np.array(label)

print(train_data.shape)
print(len(label))

print("Training a Linear LinearSVM Classifier.")
clf = LinearSVC()
clf.fit(train_data, label)
# 下面的代码是保存模型的
model_path="./"
if not os.path.exists(model_path):
    os.makedirs(model_path)
joblib.dump(clf, model_path + 'model')
#clf = joblib.load(model_path+'model')


result=clf.predict(train_data)
print(result)
print(label)

cv2.waitKey(0)
cv2.destroyAllWindows()