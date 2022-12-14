import cv2 as cv
import numpy as np
import paho.mqtt.client as mqtt
import base64
import time
import threading



local_broker_address =  "127.0.0.1"
local_broker_port = 1883
sendingVideoStream = False

def SendVideoStream (message):
    global sendingVideoStream
    cap = cv.VideoCapture(0)
    splited = message.split('/')
    origin = splited [0]
    while sendingVideoStream:
        # Read Frame
        _, frame = cap.read()
        # Encoding the Frame
        _, buffer = cv.imencode('.jpg', frame)
        # Converting into encoded bytes
        jpg_as_text = base64.b64encode(buffer)
        # Publishig the Frame on the Topic home/server
        client.publish('cameraService/'+origin+'/videoFrame', jpg_as_text)
    cap.release()




def on_message(client, userdata, message):
    global sendingVideoStream
    splited = message.topic.split('/')
    origin = splited[0]
    destination = splited[1]
    command = splited[2]

    if command == 'connectPlatform':
        print('Camera service connected by ' + origin)

        # aqui en realidad solo debería subscribirse a los comandos que llegan desde el dispositivo
        # que ordenó la conexión, pero esa información no la tiene porque el origen de este mensaje
        # es el gate. NO COSTARIA MUCHO RESOLVER ESTO. HAY QUE VER SI ES NECESARIO

        client.subscribe('+/cameraService/#')

    if command == 'takePicture':
        print('Take picture')
        cap = cv.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
        for n in range(15):
            # this loop is required to discard first frames
            ret, frame = cap.read()

        _, buffer = cv.imencode('.jpg', frame)
        # Converting into encoded bytes
        jpg_as_text = base64.b64encode(buffer)
        print('send picture')
        if origin == 'autopilotService':
            client.publish('cameraService/dashBoard/picture', jpg_as_text)
        client.publish('cameraService/' + origin + '/picture', jpg_as_text)

    if command== 'startVideoStream':
        sendingVideoStream = True
        w = threading.Thread(target=SendVideoStream, args=(message.topic,))
        w.start()

    if command == 'stopVideoStream':
        sendingVideoStream = False




client = mqtt.Client("Camera service")
client.on_message = on_message
client.connect(local_broker_address, local_broker_port)
client.loop_start()
print ('Waiting connection from DASH...')
client.subscribe('gate/cameraService/connectPlatform')
