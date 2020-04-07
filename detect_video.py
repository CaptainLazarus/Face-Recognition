import cv2
import numpy as np
import argparse
import os
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import imutils
import time

def initArgs() -> dict:
    parser = argparse.ArgumentParser()
    #parser.add_argument("-v" , "--video" , required=True , help="Path to video")
    parser.add_argument("-m" , "--model" , required=True , help="Path to Caffe pre-trained model")
    parser.add_argument("-p" , "--prototxt", required=True , help="Path to Caffe 'deploy' prototxt file")
    parser.add_argument("-c" , "--confidence" , required=False , default=0.3 , help="Threshold")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = initArgs()

    print("[INFO] Starting Model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"] , args["model"])

    print("[INFO] Starting Video Stream.........")
    
    vs = VideoStream(0).start()
    
    #For Raspberry Pi
    # vs = VideoStream(usePiCamera=True).start()    
    
    #For video files (doesnt work. Need to be modified. Work it out)
    # vs = FileVideoStream(args["video"])

    #Warm Up time
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame , width = 400)

        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame , (300,300)) , 1 , (300,300) , (104 , 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence < args["confidence"]:
                continue

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX , startY , endX , endY) = box.astype('int')

            text = "{:.2f}%".format(confidence*100)
            y = startY-10 if startY-10>10 else startY+10
            cv2.rectangle(frame , (startX,startY) , (endX , endY) , (0,0,255) , 2)
            cv2.putText(frame , text , (startX , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , (0,0,255) , 2)

        cv2.imshow("Frame" , frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    vs.stop()
