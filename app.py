"""
import Flask and create an instance of the Flask object
"""
from flask import Flask
app = Flask(__name__)


"""
import modules for car control
"""
import time
import RPi.GPIO as GPIO

"""
import flask
"""
import cv2
from camera import Camera
from datetime import datetime
from flask import Flask, request, Response, render_template

"""
import modules for object detection
"""
import os
import sys
# import cv2
import numpy as np
import tensorflow as tf

########################################################################

# pin for the buzzer
BUZZ_PIN = 12 

# pins to each motors
Motor_R1_Pin = 16
Motor_R2_Pin = 18
Motor_L1_Pin = 11
Motor_L2_Pin = 13
# pins for the control button
BTN_PIN_1 = 15
BTN_PIN_2 = 22
BTN_PIN_3 = 36
BTN_PIN_4 = 37
WAIT_TIME = 0.2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(Motor_R1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_R2_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L2_Pin, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(BTN_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_PIN_3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_PIN_4, GPIO.IN, pull_up_down=GPIO.PUD_UP)
previousStatus = None
previousTime = time.time()
currentTime = None

def stop():
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, False)


def forward():
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    time.sleep(WAIT_TIME)
    stop()


def backward():
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, True)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, True)
    time.sleep(WAIT_TIME)
    stop()


def turnRight():
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    time.sleep(WAIT_TIME)
    stop()

def turnLeft():
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, False)
    time.sleep(WAIT_TIME)
    stop()

########################################################################

pitches = [262, 294, 330, 349, 392, 440, 493, 523]
GPIO.setup(BUZZ_PIN, GPIO.OUT)

pwm = GPIO.PWM(BUZZ_PIN, pitches[0])
pwm.start(0)

def play(pitch, intv):
    pwm.ChangeFrequency(pitch)
    time.sleep(intv)

########################################################################

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
# print(PATH_TO_LABELS)
# Number of classes the object detector can identify
NUM_CLASSES = 90

# Set up camera constants
IM_WIDTH = 640    # Use smaller resolution for
IM_HEIGHT = 480   # slightly faster framerate

## Load the label map.
# Label maps map indices to category names, so that when the convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


object = False
def obj_detect(frame):
    # success, frame = camera.video.read()
    # if success:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.01)

    global object  
    cs = np.squeeze(classes).astype(np.int32)
    sc = np.squeeze(scores)

    for i in range(int(num[0])):
        if sc[i] > 0.5:
            cx = (boxes[0][i][1] + boxes[0][i][3]) / 2
            cy = (boxes[0][i][0] + boxes[0][i][2]) / 2
            frame = cv2.circle(frame, (int(cx*IM_WIDTH),int(cy*IM_HEIGHT)), radius=5, color=(0,0,255), thickness=-1)
            print(cx)
            if cx > 0.3 and cx < 0.6:
                # print("alert")
                object=True
                pwm.ChangeDutyCycle(50)
                play(262,1)
                pwm.ChangeDutyCycle(0)
                time.sleep(5)
                object=False
        object=False

@app.route('/alert')
def alert():
    global object
    input1 = GPIO.input(BTN_PIN_1)
    input2 = GPIO.input(BTN_PIN_2)
    input3 = GPIO.input(BTN_PIN_3)
    input4 = GPIO.input(BTN_PIN_4)
    if input1 == GPIO.LOW:
        print('forward')
        forward()

    if input2 == GPIO.LOW:
        print('right')
        turnRight()

    if input3 == GPIO.LOW:    
        print('left')
        turnLeft()

    if input4 == GPIO.LOW:    
        print('backward')
        backward()
    return render_template(
        "alert.html",
        object=object,
    )

"""
Add functions that returns the web page 
and use Flask's app.route decorator to map the URL route to each function
"""
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        command = list(request.form.keys())
        command = command[0]
        if command == 'UP':
            """
            this section that moves car forward
            """
            forward()
            print('UP')
        elif command == 'LEFT':
            """
            this section that moves car turn left
            """
            turnLeft()
            print('LEFT')
        elif command == 'RIGHT':
            """
            this section that moves car turn right
            """
            turnRight()
            print('RIGHT')
        elif command == 'DOWN':
            """
            this section that moves car turn backward
            """
            backward()
            print('DOWN')
    # user render_template to load a template
    # and apply the named values
    return render_template(
        "stream.html",
        date=datetime.now(),
    )

"""
Get data from a webcam
""" 
def genVideo(camera):
    while True:
        # print("Generatre Video")
        image = camera.getFrame()
        try:
            obj_detect(image)
        except:
            print("can not obj-detect")
        try:
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            return ' '

@app.route('/video_feed')
def video_feed():
    return Response(genVideo(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

