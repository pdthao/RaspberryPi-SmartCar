import cv2

class Camera(object):
    def __init__(self):
        if cv2.__version__.startswith('2'):
            PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        elif cv2.__version__.startswith('3') or cv2.__version__.startswith('4'):
            PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        
        # This is for rpi camera 
        # self.video = cv2.VideoCapture(0, cv2.CAP_V4L)
        self.video = cv2.VideoCapture(0)
        
        # cv2.namedWindow("preview") # check what is taken by the cam
        
        self.video.set(PROP_FRAME_WIDTH, 320)
        self.video.set(PROP_FRAME_HEIGHT, 240)
        
    def __del__(self):
        self.video.release()
        # cv2.destroyWindow("preview")

    def getFrame(self):
        success, image = self.video.read()
        if success:
            return image
        print("Can not open camera")    
