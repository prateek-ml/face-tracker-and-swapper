import cv2
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker

class Camera(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        self._faceTracker = FaceTracker()
        self._shoulddrawRects = False

    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            #Swapping faces in a camera feed
            self._faceTracker.update(frame)
            tracked_faces = self._faceTracker.faces
            face_rects = []
            for tf in tracked_faces:
                face_rects.append(tf.faceRect)
            rects.swapRects(frame, frame, face_rects)
            
            if self._shoulddrawRects:
                self._faceTracker.drawRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    def onKeypress(self, keycode):
        """Handle a keypress
        
        space    -> Take a screenshot.
        tab      -> Start/stop recording a screencast
        escape   -> Quit
        x        -> Start/stop drawing rectangles
        """
        if keycode == 32: #space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: #tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.mp4')
            
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: #x
            self._shoulddrawRects = not self._shoulddrawRects

        elif keycode == 27: #escape
            self._windowManager.destroyWindow()
        



if __name__=="__main__":
    Camera().run()
    print(cv2.__file__)



