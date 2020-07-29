# Face Tracker and Swapper

In this repo, I have created a face tracker and swapper using the HAAR Cascade Classifier provided by the OpenCV library. It's function is to track faces and its features from the camera feed, (you can use the one I created in the OpenCV-Boilerplate) and if multiple faces are detected in the image, swap those faces with each other in a circular manner. 

## How To Use
Simply clone this repo, and in your terminal run the <code>camera.py</code> script. Your webcam will be activated and the camera feed will display the output.

- **Activating Face Detector** : Press the 'X' key and wait for a moment, the face detector feature is turned on. You will now see a few rectangles around your face and eyes.
- **Activating Face Swapper** : Press the 'Q' key and if the camera detects more than 1 face, your faces will be swapped. 


You can find the OpenCV boilerplate [here](https://www.github.com/prateek-ml/opencv-boilerplate "OpenCV Boilerplate by Prateek Bhardwaj")
