from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from itertools import combinations
import math
from imutils.video import WebcamVideoStream

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
prototxt = "deploy.prototxt"
weights_path = "SSD.caffemodel"
faceNet=cv2.dnn.readNet(prototxt,weights_path)
maskNet = load_model("new_improved_model.h5")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(300,300))
            face=img_to_array(face)    
            #face=preprocess_input(face)    
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        if len(faces)==0:
            cv2.putText(frame, "No Person", (600,63), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return (locs,preds)


def main():
    cap = cv2.VideoCapture(0)
    #vs = WebcamVideoStream(src=0).start()
    """fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0"""

    while True:
        ret, frame = cap.read()
        #frame = vs.read()
        frame = imutils.resize(frame, width=786)
        #total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        centroid_dict = dict()
        objects = tracker.update(rects)
        
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)
            text = "ID: {}".format(objectId+1)
            #
            for (box,pred) in zip(locs,preds):
                (startX,startY,endX,endY)=box
                cv2.putText(frame,text,(startX+117,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.putText(frame, "Maintain Social Distance", (5,63), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            """else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)"""


        """fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)"""

        #(locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
    
        #loop over the detected face locations and their corrosponding loactions
    
        for (box,pred) in zip(locs,preds):
            (startX,startY,endX,endY)=box
            #(mask,withoutMask)=pred
        
            #determine the class label and color we will use to draw the bounding box and text
            label='Mask' if pred==0 else 'No Mask'
            color=(0,255,0) if label=='Mask' else (0,0,255)
        
            #display the label and bounding boxes (x1, y1-5)
            cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            #cv2.putText(frame,label,(x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        cv2.putText(frame, "Made with love by Somu", (5, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
 
        
        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    #vs.stop()

main()
