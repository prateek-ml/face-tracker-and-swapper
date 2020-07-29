import cv2
import numpy as np
import rects
import utils

class Face(object):
    """Data on facial features: face, eyes, nose, mouth."""

    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.smileRect = None
    
class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth"""

    def __init__(self, scaleFactor = 1.3, minNeighbors = 2, flags=cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []

        self._faceClassifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
        self._LeyeClassifier = cv2.CascadeClassifier('cascades/haarcascade_lefteye_2splits.xml')
        self._ReyeClassifier = cv2.CascadeClassifier('cascades/haarcascade_righteye_2splits.xml')
        self._smileClassifier = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')
    
    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces


    # Detecting a single face feature by using the classifier on an image slice
    def _detectOneObject(self, classifier, image, rect, imageSizetoMinSizeRatio):
        x, y, w, h = rect

        x = int(x)
        y = int(y)

        w = int(w)
        h= int(h)

        minSize = utils.DividedWidthHeight(image, imageSizetoMinSizeRatio)
        
        subImage = image[y:y+h, x:x+w]

        subRects = classifier.detectMultiScale(
            subImage, self.scaleFactor, self.minNeighbors, self.flags, minSize)

        if len(subRects) == 0:
            return None
        
        subX, subY, subW, subH = subRects[0]

        return (x+subX, y+subY, w+w+subW, h+subH)

    def update(self, image):
        """Update the tracked facial features."""

        self._faces = []

        #Creating an equalized, grayscale variant of the image
        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)
        
        minSize = utils.DividedWidthHeight(image,8)

        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags)
        
        if faceRects is not None:
            for faceRect in faceRects:

                face = Face()
                face.faceRect = faceRect

                x, y, w, h = faceRect


                #Seek an eye in the upper-left part of the face
                searchRect = (x+w/7, y, w*2/7, h/2)
                face.leftEyeRect = self._detectOneObject(
                    self._LeyeClassifier, image, searchRect, 64
                )

                #Seek an eye in the upper-right part of the face
                searchRect = (x+w/4, y+h/4, w*2/7, h/2)
                face.rightEyeRect = self._detectOneObject(
                    self._ReyeClassifier, image, searchRect, 64
                )

                # Seek a mouth in the lower-middle part of the face.
                searchRect = (x+w/6, y+h*2/3, w*2/3, h/3)
                face.smileRect = self._detectOneObject(
                self._smileClassifier, image, searchRect, 16)
                self._faces.append(face)
    
    def drawRects(self, image):
        """Draw rectangles around the tracked facial feature"""

        #White rectangles in a grayscale image
        if utils.isGray(image):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
            smileColor = 255
        
        else:
            faceColor = (255, 255, 255)  #White
            leftEyeColor = (0, 0, 255)   #Red
            rightEyeColor = (255, 0, 0)  #Blue
            smileColor = (0,255,0)       #Green
        
        for face in self._faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            rects.outlineRect(image, face.smileRect, smileColor)
        
