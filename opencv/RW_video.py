import numpy as np
import os
import cv2
def videoFromwebcame():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Webcam Video', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release() 
    cv2.destroyAllWindows()  
def videoFromFile():
    root = os.getcwd()
    filename = 'demoimages//VID_20230308180159.mp4'
    videoPath = os.path.join(root, filename)
    cap = cv2.VideoCapture(videoPath)  
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Video File', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release() 
    cv2.destroyAllWindows()
def writevideofromfile():
    root = os.getcwd()
    filename = 'demoimages//VID_20230308180159.mp4'
    videoPath = os.path.join(root, filename)
    cap = cv2.VideoCapture(videoPath)  
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 480)) 

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame.")
            break

        out.write(frame)  
        cv2.imshow('Video File', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release() 
    out.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #videoFromwebcame()
    #videoFromFile()
    writevideofromfile()