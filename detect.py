import cv2
import numpy as np
import argparse
import os
from imutils import paths

def initArgs() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i" , "--image" , required=True , help="Path to input image directory")
    parser.add_argument("-m" , "--model" , required=True , help="Path to Caffe pre-trained model")
    parser.add_argument("-p" , "--prototxt", required=True , help="Path to Caffe 'deploy' prototxt file")
    parser.add_argument("-c" , "--confidence" , required=False , default=0.3 , help="Threshold")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = initArgs()
    print("[INFO] Loading Model")
    
    net = cv2.dnn.readNetFromCaffe(args["prototxt"] , args["model"])
    
    ipaths = list(paths.list_images(args["image"]))

    for i in ipaths:
        #Pre processing
        image = cv2.imread(i)
        (h,w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image , (300,300)) , 1.0 , (300,300) , (104 , 177 , 123))
        
        #Detections
        print("[INFO] Detecting faces")
        net.setInput(blob)
        detections = net.forward()

        #Drawing bounding boxes
        for j in range(detections.shape[2]):
            confidence = detections[0,0,j,2]

            if confidence > args["confidence"]:
                box = detections[0,0,j,3:7] * np.array([w,h,w,h])
                (startX , startY , endX , endY) = box.astype("int")

                text = "{:.2f}%".format(confidence*100)
                y = startY-10 if startY-10 > 10 else startY+10
                cv2.putText(image , text , (startX,y) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , (0,0,255) , 2)
                cv2.rectangle(image , (startX,startY) , (endX,endY) , (0,0,255) , 2)
        print("{} : {}".format(i , text ) )
        cv2.imshow("Output" , cv2.resize(image , (600,600) ))
        cv2.waitKey(0)