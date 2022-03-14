#!/usr/bin/env python

import cv2
import os
import sys, getopt
import numpy as np
import signal
from edge_impulse_linux.image import ImageImpulseRunner
from gpiozero import Button, Servo
from time import sleep

button = Button(2)
servo1 = Servo(17)
servo2 = Servo(18)
runner = None

show_camera = False

if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify-image.py <path_to_model.eim> <path_to_image.jpg>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 1:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)
    
    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            if len(args)>= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args)<= 1 and len(port_ids)> 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
               
            else:
                raise Exception("Couldn't initialize selected camera.")
            
            while True:
                servo1.mid()
                sleep(2)
                servo2.mid()
                sleep(2)
                
                button.wait_for_press()
                print("Button Pushed")
                ret, img = camera.read() 

                # imread returns images in BGR format, so we need to convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(img)

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                if show_camera == True:
                    cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                res = runner.classify(features)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    high_score = 0
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                        if score > high_score:
                           high_score = score
                           winner = label
                           
                    print('', flush=True)
                    print(winner)
                    if winner == "paper":
                        servo1.min()
                        sleep(2)
                    elif winner == "plastic":
                        servo1.max()
                        sleep(2)
                    elif winner == "metal":
                        servo2.min()
                        sleep(2)
                    elif winner == "trash":
                        servo2.max()
                        sleep(2)
                elif "bounding_boxes" in res["result"].keys():
                    print('The model you are using is producing bounding boxes. However, bounding boxes not supported')
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
