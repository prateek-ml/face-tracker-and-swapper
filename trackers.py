import cv2
import utils


class Face(object):
    """Data on facial features: face, eyes, nose, mouth."""

    def __init__(self):
        self.faceRect = None
        self.EyeRect = None
    
class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth"""

    def __init__(self, scaleFactor = 1.3, minNeighbors = 2, flags=cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []

        self._faceClassifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        self._eyeClassifier = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    
    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces

    def update(self, image):
        """Update the tracked facial features."""

        self._faces = []

        #Creating an equalized, grayscale variant of the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        minSize = (120,120)

        faceRects = self._faceClassifier.detectMultiScale(
            gray, self.scaleFactor, self.minNeighbors, self.flags, minSize)
        
        if faceRects is not None:
            for faceRect in faceRects:

                face = Face()
                face.faceRect = faceRect

                (x,y,w,h) = faceRect

                #Detect eyes in the face
                minSize = (40,40)
                eye_region = gray[y:y+h, x:w+w]
                face.eyeRects = self._eyeClassifier.detectMultiScale(
                    eye_region, self.scaleFactor, self.minNeighbors, self.flags, minSize
                )

                self._faces.append(face)
    
    def drawRects(self, image):
        """Draw rectangles around the tracked facial feature"""

        #White rectangles in a grayscale image
        if utils.isGray(image):
            faceColor = 255
        
        else:
            faceColor = (255, 255, 255)  #White
            
        
        for face in self._faces:
            (x, y, w, h) = face.faceRect
            cv2.rectangle(image, (x,y), (x+w, y+h), faceColor, 2)

            for eyeRect in face.eyeRects:
                (ex, ey, ew, eh) = eyeRect
                cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)


        


