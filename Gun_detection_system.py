import numpy as np
import cv2
import imutils
import datetime

# Load the gun cascade classifier from the XML file
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Variable to store the first frame for background reference
firstFrame = None

# Variable to check if a gun exists in the video feed
gun_exist = None

# Start an infinite loop to continuously capture frames from the webcam
while True:
    # Read the current frame from the webcam
    ret, frame = camera.read()
    
    # Resize the frame for faster processing and consistent dimensions
    frame = imutils.resize(frame, width=500)
    
    # Convert the frame to grayscale for the cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect guns in the frame using the cascade classifier
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    
    # If guns are detected, set the gun_exist flag to True
    if len(gun) > 0:
        gun_exist = True
    
    # Loop through all detected guns and draw rectangles around them
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) in both grayscale and color frames
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    
    # Set the first frame for background reference if it hasn't been set yet
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # Display the current frame in a window called "security feed"
    cv2.imshow("security feed", frame)
    
    # Wait for 1 millisecond for a key press and check if 'q' was pressed to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# After the loop ends, print whether guns were detected or not
if gun_exist:
    print("Guns detected")
else:
    print("No guns detected")

# Release the webcam and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
