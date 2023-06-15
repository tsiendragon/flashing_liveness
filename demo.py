import cv2
import numpy as np
import argparse
import time
import imutils
from imutils.video import VideoStream
import os



ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--detector", type=str, required=True,
#	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以尝试不同的索引值





# 设置保存路径
savepath="./data"

if not os.path.exists(savepath):
    os.makedirs(savepath)



def preprocess(fa):
    fa = cv2.resize(fa, (224, 224))
#   fa = np.transpose(face, (2, 0, 1))
    fa = fa[74:149,74:149,:]
    return fa


#load the face detector

print("[INFO] loading face detector...")
protoPath = os.path.join("./face_detector","deploy.prototxt")
modelPath = os.path.join("./face_detector","res10_300x300_ssd_iter_140000.caffemodel")
net=cv2.dnn.readNetFromCaffe(protoPath,modelPath)

# ...

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()



whether_break=False

while True:

    Detection = False
    time.sleep(1.0)
    time_slot = 0
    start_time = 0
    save_original = False
    outside_color_index = 0
    Capture_color = []
    color_name = ["white", "blue", "green", "red"]


    if whether_break:
        break

    whether_break=False

    while True:



        ret, frame = cap.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=1800)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        #frame.shape=[450,600,3]

        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                face = frame[startY:endY, startX:endX]
    #            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
    #            print(startX, startY,endX, endY)
                # Get the shape of the captured frame

                height, width, channels = frame.shape

                # Create a blue NumPy array with the same shape
                mask = np.zeros((height, width, channels), dtype=np.uint8)

                x=630
                y=380
                x_end=x+500
                y_end=y+700

                cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)

                if not Detection:
                    if startX>=x and startY>=y and endX<=x_end and endY<=y_end:
                        cv2.rectangle(frame, (x,y), (x_end,y_end), (0,255,0), 2)

                        if start_time==0:
                            start_time=time.time()
                        elif time.time()-start_time>1:

                            if not save_original:
                                file_name = os.path.join(savepath, "original.png")
                                cv2.imwrite(file_name, face)
                                original_face=face
                                print("[INFO] save original face")
                                save_original=True

                            if outside_color_index<=2:
                                mask[:, :, outside_color_index] = 255
                                mask[y:y_end,x:x_end] = frame[y:y_end,x:x_end]
                                frame=mask

                                if time_slot==0:
                                    time_slot = time.time()
                                elif time.time()-time_slot >= 2:
                                    outside_color_index+=1
                                    time_slot=time.time()
                                elif time.time()-time_slot>1 and len(Capture_color)==outside_color_index:
                                    file_name = os.path.join(savepath, "{}.png".format(color_name[outside_color_index]))
                                    cv2.imwrite(file_name, face)
                                    Capture_color.append(face)
                                    print("[INFO] save a {} face".format(color_name[outside_color_index]))

                    if len(Capture_color)==3:
                        count=0
                        original_face=preprocess(original_face)
                        for i,face in enumerate(Capture_color):
                            face=preprocess(face)
                            v=np.mean(face[:,:,i]-original_face[:,:,i])
                            print(v)
                            if v<70:
                                count+=1

                        if count==3:
                            res = True
                        else:
                            res = False

                        Detection=True

                else:
                    if res:
                        cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
                        cv2.putText(frame, "Real", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)
                        cv2.putText(frame, "Fake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        # 显示画面
        cv2.imshow("Face Capture", frame)


        # 检测键盘按键，按下 "q" 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            whether_break=True
            break

        if cv2.waitKey(1) & 0xFF == ord('r'):
            print("[INFO] new round")
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()