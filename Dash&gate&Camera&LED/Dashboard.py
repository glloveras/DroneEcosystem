import base64
import json
import os
import subprocess
import time

import tkinter as tk
import cv2 as cv
import numpy as np
from PIL import ImageTk, Image
from tkinter import scrolledtext, font, W, CENTER
from  tkinter import ttk

import paho.mqtt.client as mqtt
from djitellopy import Tello

import math
from tkinter import Menu
import tensorflow as tf
import os
# importar libreria para ver el funcionamiento de la red
from tensorflow.keras.callbacks import TensorBoard
# importar libreria para poder modificar las imagenes
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array


def calc_pixels(tupla_1, tupla_2):
    pixels = math.sqrt(math.pow((tupla_1[0]-tupla_2[0]), 2) + math.pow((tupla_1[1]-tupla_2[1]), 2))
    return pixels


master = tk.Tk()
client = mqtt.Client('Dashboard')
global_broker_address ="127.0.0.1"
global_broker_port = 1884

# treatment of messages received from gate through the global broker

def on_message(client, userdata, message):
    global panel
    global lbl
    global table

    splited = message.topic.split('/')
    origin = splited[0]
    destination = splited[1]
    command = splited[2]


    if origin == "cameraService":

        if command == "videoFrame":
            img = base64.b64decode(message.payload)
            # converting into numpy array from buffer
            npimg = np.frombuffer(img, dtype=np.uint8)
            # Decode to Original Frame
            img = cv.imdecode(npimg, 1)
            # show stream in a separate opencv window
            cv.imshow("Stream", img)
            cv.waitKey(1)
        if command == 'picture':
            print('show picture')
            img = base64.b64decode(message.payload)
            # converting into numpy array from buffer
            npimg = np.frombuffer(img, dtype=np.uint8)
            # Decode to Original Frame
            cv2image = cv.imdecode(npimg, 1)
            gray = cv.cvtColor(cv2image, cv.COLOR_BGR2GRAY)
            print(detector_ball)
            if detector_ball:
                ball = detector_ball.detectMultiScale(gray, 1.35, 20)
                for (x, y, w, h) in ball:
                    cv.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif detector_face and detector_body:
                print('detect face')
                face = detector_face.detectMultiScale(gray, 1.2, 5)
                body = detector_body.detectMultiScale(gray, 1.2, 5)
                for (x, y, w, h) in face:
                    cv.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (x, y, w, h) in body:
                    cv.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif detector_2:
                gray = cv.resize(gray, (200, 200), interpolation=cv.INTER_CUBIC)
                gray = np.array(gray).astype(float) / 255
                img = img_to_array(gray)
                img = np.expand_dims(img, axis=0)
                predict = detector_2.predict(img)
                predict = predict[0][0]
                if predict > 0.98:
                    cv.putText(cv2image, "Cat or dog detected", (200, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))
            dim = (300, 300)
            # resize image
            cv2image = cv.resize(cv2image, dim, interpolation=cv.INTER_AREA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.imgtk = imgtk
            panel.configure(image=imgtk)
    if origin == "autopilotService":
        if (command == "droneAltitude"):
            answer = str(message.payload.decode("utf-8"))
            lbl['text'] = answer[:5]
        if (command == "droneHeading"):
            answer = str(message.payload.decode("utf-8"))
            lbl['text'] = answer[:5]
        if (command == "droneGroundSpeed"):
            answer = str(message.payload.decode("utf-8"))
            lbl['text'] = answer[:5]

        if (command  == "dronePosition"):
            positionStr = str(message.payload.decode("utf-8"))
            position = positionStr.split('*')
            latLbl['text'] = position[0]
            lonLbl['text'] = position[1]

        if (command  == "droneBattery"):
            answer = str(message.payload.decode("utf-8"))
            lbl['text'] = answer[:5]

    if origin == "dataService":
        if  (command == "storedPositions"):
            # receive the positions stored by the data service
            data = message.payload.decode("utf-8")
            # converts received string to json
            dataJson = json.loads(data)
            cont = 0
            for dataItem in dataJson:
                table.insert(parent='', index='end', iid=cont, text='',
                        values=(dataItem['time'], dataItem['lat'], dataItem['lon']))
                cont = cont + 1

            table.pack()

    if origin == "radiationService":
        if (command == "Radiation"):
            answer = str(message.payload.decode("utf-8"))
            lbl['text'] = answer[:5] + ' mSv'



client.on_message = on_message

# |--DASHBOARD master frame ----------------------------------------------------------------------------------|
# |                                                                                                           |
# |  |---connection frame--------------------------------------------------------------------------------|    |
# |  |---------------------------------------------------------------------------------------------------|    |
# |                                                                                                           |
# |  |---top frame---------------------------------------------------------------------------------------|    |
# |  |                                                                                                   |    |
# |  |   |--Autopilot control label frame ----------------------------|  |--LEDs control label frame--|  |    |
# |  |   |                                                            |  |                            |  |    |
# |  |   |  |--Arm/disarm frame -----------------------------------|  |  |----------------------------|  |    |
# |  |   |  |------------------------------------------------------|  |                                  |    |
# |  |   |                                                            |                                  |    |
# |  |   |  |--bottom frame ---------------------------------------|  |                                  |    |
# |  |   |  |                                                      |  |                                  |    |
# |  |   |  |  |-Autopilot get frame--|  |-Autopilot set frame -|  |  |                                  |    |
# |  |   |  |  |----------------------|  |----------------------|  |  |                                  |    |
# |  |   |  |                                                      |  |                                  |    |
# |  |   |  |------------------------------------------------------|  |                                  |    |
# |  |   |                                                            |                                  |    |
# |  |   |------------------------------------------------------------|                                  |    |
# |  |---------------------------------------------------------------------------------------------------|    |
# |                                                                                                           |
# |  |---camera control label frame----------------------------------------------------------------------|    |
# |  |                                                                                                   |    |
# |  |   |--- Take picture frame -----------|            |--- Video stream frame -----------|            |    |
# |  |   |                                  |            |                                  |            |    |
# |  |   |----------------------------------|            |----------------------------------|            |    |
# |  |---------------------------------------------------------------------------------------------------|    |
# |                                                                                                           |
# |-----------------------------------------------------------------------------------------------------------|



# Connection frame ----------------------
connected = False
connectionFrame = tk.Frame (master)
connectionFrame.pack(fill = tk.X)

def connectionButtonClicked():
    global connected
    global client
    global detector_ball
    detector_ball = False
    global detector_2
    detector_2 = False
    global detector_body
    detector_body = False
    global detector_face
    detector_face = False

    if not connected:
        connectionButton['text'] = "Disconnect"
        connectionButton['bg'] = "green"
        connected = True
        client.connect(global_broker_address,  global_broker_port)
        client.publish("dashBoard/gate/connectPlatform")
        client.loop_start()
        client.subscribe("+/dashBoard/#")
        print('Connected with drone platform')

        topFrame.pack(fill=tk.X)
        cameraControlFrame.pack(padx=20, pady=20);

    else:
        print('Disconnect')
        connectionButton['text'] = "Connect with drone platform"
        connectionButton['bg'] = "red"
        connected = False
        topFrame.pack_forget()
        ledsControlFrame.pack_forget()
        cameraControlFrame.pack_forget()

connectionButton = tk.Button(connectionFrame, text="Connect with drone platform", width = 50, bg='red', fg="white", command=connectionButtonClicked)
connectionButton.grid(row = 0, column = 0, padx=60, pady=20)
# top frame -------------------------------------------
topFrame = tk.Frame (master)


# Autopilot control label frame ----------------------
autopilotControlFrame = tk.LabelFrame(topFrame, text="Autopilot control", padx=5, pady=5)
autopilotControlFrame.pack(padx=20, side = tk.LEFT);

# Arm/disarm frame ----------------------
armDisarmFrame = tk.Frame (autopilotControlFrame)
armDisarmFrame.pack(padx=20)

armed = False
def armDisarmButtonClicked():
    global armed

    if not armed:
            armDisarmButton['text'] = "Disarm drone"
            armDisarmButton['bg'] = "green"
            armed = True
            client.publish("dashBoard/autopilotService/armDrone")

    else:
            armDisarmButton['text'] = "Arm drone"
            armDisarmButton['bg'] = "red"
            armed = False
            client.publish("dashBoard/autopilotService/disarmDrone")



armDisarmButton = tk.Button(armDisarmFrame, text="Arm drone", bg='red', fg="white",  width = 90, command=armDisarmButtonClicked)
armDisarmButton.grid(column=0, row=0,  pady = 5)

# bottomFrame frame ----------------------
bottomFrame = tk.Frame (autopilotControlFrame)
bottomFrame.pack(padx=20)

# Autopilot get frame ----------------------
autopilotGet = tk.Frame (bottomFrame)
autopilotGet.pack(side = tk.LEFT, padx=20)

v1 = tk.StringVar()
s1r1= tk.Radiobutton(autopilotGet,text="Altitude", variable=v1, value=1).grid(column=0, row=0, columnspan = 5, sticky=tk.W)
s1r2= tk.Radiobutton(autopilotGet,text="Heading", variable=v1, value=2).grid(column=0, row=1, columnspan = 5, sticky=tk.W)
s1r3= tk.Radiobutton(autopilotGet,text="Ground Speed", variable=v1, value=3).grid(column=0, row=2, columnspan = 5, sticky=tk.W)
s1r4= tk.Radiobutton(autopilotGet,text="Battery level", variable=v1, value=4).grid(column=0, row=3, columnspan = 5, sticky=tk.W)
v1.set(1)

def autopilotGetButtonClicked():
    if v1.get() == "1":
        client.publish("dashBoard/autopilotService/getDroneAltitude")
    elif v1.get() == "2":
        client.publish("dashBoard/autopilotService/getDroneHeading")
    elif v1.get() == "3":
        client.publish("dashBoard/autopilotService/getDroneGroundSpeed")
    else:
        client.publish("dashBoard/autopilotService/getDroneBattery")

autopilotGetButton = tk.Button(autopilotGet, text="Get", bg='red', fg="white", width = 10, height=5, command=autopilotGetButtonClicked)
autopilotGetButton.grid(column=5, row=0, columnspan=2, rowspan = 3, padx=10)

lbl = tk.Label(autopilotGet, text=" ", width = 10, borderwidth=2, relief="sunken")
lbl.grid(column=7, row=1,  columnspan=2 )

# Autopilot set frame ----------------------
autopilotSet = tk.Frame (bottomFrame)
autopilotSet.pack( padx=20)



def takeOffButtonClicked():
    client.publish("dashBoard/autopilotService/takeOff", metersEntry.get() )

takeOffButton = tk.Button(autopilotSet, text="Take Off", bg='red', fg="white",  width = 10, command=takeOffButtonClicked)
takeOffButton.grid(column=0, row=1, columnspan=2, sticky=tk.W)

to = tk.Label(autopilotSet, text="to")
to.grid(column=2, row=1)
metersEntry = tk.Entry(autopilotSet, width = 10)
metersEntry.grid(column=3, row=1,  columnspan=2 )
meters = tk.Label(autopilotSet, text="meters")
meters.grid(column=5, row=1)



lat = tk.Label(autopilotSet, text="lat")
lat.grid(column=2, row=2,  columnspan=2,padx = 5 )

lon = tk.Label(autopilotSet, text="lon")
lon.grid(column=4, row=2,  columnspan=2,padx = 5 )

def getPositionButtonClicked():
    client.publish("dashBoard/autopilotService/getDronePosition" )


getPositionButton = tk.Button(autopilotSet, text="Get Position", bg='red', fg="white",  width = 10,  command=getPositionButtonClicked)
getPositionButton.grid(column=0, row=3, pady = 5, sticky=tk.W)

latLbl = tk.Label(autopilotSet, text=" ", width = 10, borderwidth=2, relief="sunken")
latLbl.grid(column=2, row=3,  columnspan=2,padx = 5 )

lonLbl = tk.Label(autopilotSet, text=" ", width = 10, borderwidth=2, relief="sunken")
lonLbl.grid(column=4, row=3,  columnspan=2,padx = 5 )

def goToButtonClicked():
    position = str (goTolatEntry.get()) + '*' + str(goTolonEntry.get())
    client.publish("dashBoard/autopilotService/goToPosition", position)



goToButton = tk.Button(autopilotSet, text="Go To", bg='red', fg="white",  width = 10,  command=goToButtonClicked)
goToButton.grid(column=0, row=4, pady = 5, sticky=tk.W)

goTolatEntry = tk.Entry(autopilotSet, width = 10)
goTolatEntry.grid(column=2, row=4,  columnspan=2,padx = 5 )

goTolonEntry = tk.Entry(autopilotSet, width = 10)
goTolonEntry.grid(column=4, row=4,  columnspan=2,padx = 5 )

def returnToLaunchButtonClicked():
    client.publish("dashBoard/autopilotService/returnToLaunch")




returnToLaunchButton = tk.Button(autopilotSet, text="Return To Launch", bg='red', fg="white",  width = 40, command=returnToLaunchButtonClicked)
returnToLaunchButton.grid(column=0, row=5,  pady = 5, columnspan=6, sticky=tk.W)


def openWindowToShowRecordedPositions():
    # Open a new small window to show the positions timestamp to be received from the data service
    global newWindow
    global table
    newWindow = tk.Toplevel(master)


    newWindow.title("Recorded positions")

    newWindow.geometry("400x400")
    table = ttk.Treeview(newWindow)

    table['columns'] = ('time', 'latitude', 'longitude')

    table.column("#0", width=0, stretch=tk.NO)
    table.column("time", anchor=tk.CENTER, width=150)
    table.column("latitude", anchor=tk.CENTER, width=80)
    table.column("longitude", anchor=tk.CENTER, width=80)


    table.heading("#0", text="", anchor=tk.CENTER)
    table.heading("time", text="Time", anchor=tk.CENTER)
    table.heading("latitude", text="Latitude", anchor=tk.CENTER)
    table.heading("longitude", text="Longitude", anchor=tk.CENTER)

    # requiere the stored positions from the data service
    client.publish("dashBoard/dataService/getStoredPositions")

    closeButton = tk.Button(newWindow, text="Close", bg='red', fg="white", command=closeWindowToShowRecordedPositions).pack()



def closeWindowToShowRecordedPositions ():
    global newWindow
    newWindow.destroy()

showRecordedPositionsButton = tk.Button(autopilotSet, text="Show recorded positions", bg='red', fg="white",  width = 40, command=openWindowToShowRecordedPositions)
showRecordedPositionsButton.grid(column=0, row=6,  pady = 5, columnspan=6, sticky=tk.W)

# LEDs control frame ----------------------
ledsControlFrame = tk.LabelFrame(topFrame, text="LEDs control", padx=5, pady=5)
ledsControlFrame.pack(padx=20, pady=20);

v3 = tk.StringVar()
s1r7= tk.Radiobutton(ledsControlFrame,text="LED sequence START/STOP", variable=v3, value=1).grid(column=2, row=2, columnspan = 3)
s1r8= tk.Radiobutton(ledsControlFrame,text="LED sequence for N seconds", variable=v3, value=2).grid(column=2, row=3, columnspan = 3)

seconds = tk.Entry(ledsControlFrame, width = 5)
seconds.grid(column=5, row=3, columnspan = 3)
v3.set(1)

lEDSequence = False;

def LEDControlButtonClicked():
    global E1
    global lEDSequence
    if v3.get() == "1":
        if not lEDSequence:
            ledControlButton['text'] = "Stop"
            ledControlButton['bg'] = "green"
            lEDSequence = True
            print ('Start LEDs sequence')
            client.publish("dashBoard/LEDsService/startLEDsSequence")

        else:
            ledControlButton['text'] = "Start"
            ledControlButton['bg'] = "red"
            lEDSequence = False
            print('Stop LEDs sequence')
            client.publish("dashBoard/LEDsService/stopLEDsSequence")

    if v3.get() == "2":
            print('LEDs sequence for N seconds')
            client.publish("dashBoard/LEDsService/LEDsSequenceForNSeconds", seconds.get())


ledControlButton = tk.Button(ledsControlFrame, text="Start", bg='red', fg="white",  width = 10, height = 3, command=LEDControlButtonClicked)
ledControlButton.grid(column=8, row=1,  padx = 5, columnspan=4, rowspan = 3)

# Radiation control frame ----------------------
radiationControlFrame = tk.LabelFrame(topFrame, text="Radiation control", padx=5, pady=5)
radiationControlFrame.pack(padx=20, pady=20)

v4 = tk.StringVar()
s1r9= tk.Radiobutton(radiationControlFrame,text="Radiation value", variable=v4, value=4).grid(column=0, row=1, columnspan = 5, sticky=tk.W)
v4.set(1)

def radiationGetButtonClicked():
    if v4.get() == "1":
        client.publish("dashBoard/radiationService/getRadiation")

radiationGetButton = tk.Button(radiationControlFrame, text="Get", bg='red', fg="white", width = 10, height=3, command=radiationGetButtonClicked)
radiationGetButton.grid(column=5, row=0, columnspan=2, rowspan = 3, padx=10)

lbl = tk.Label(radiationControlFrame, text=" ", width = 10, borderwidth=2, relief="sunken")
lbl.grid(column=7, row=1,  columnspan=2 )

# flightplan open map frame ----------------------
flightplanControlFrame = tk.LabelFrame(topFrame, text="Flightplan Map", padx=5, pady=5)
flightplanControlFrame.pack(padx=20, pady=20)

def openFlightplanButtonClicked():
    print('Opening Flightplan map')
    client.publish("dashBoard/flightplanService/openFlightplan")
    client.subscribe("flightplanService/#")

flightplanOpenButton = tk.Button(flightplanControlFrame, text="OPEN", bg='red', fg="white", width = 10, height=3, command=openFlightplanButtonClicked)
flightplanOpenButton.grid(column=5, row=0, columnspan=2, rowspan = 3, padx=10)

# Camera control label frame ----------------------
cameraControlFrame = tk.LabelFrame(master, text="Camera control", padx=5, pady=5)


takePictureFrame = tk.Frame (cameraControlFrame)
takePictureFrame.pack(side = tk.LEFT)

def train():
    text_1 = input_1_text.get()
    text_2 = input_2_text.get()
    epoch = input_3_text.get()
    print(text_1, text_2, epoch)
    ## Almacenamiento de imagnes
    train_path = text_1
    vali_path = text_2
    train_list = os.listdir(train_path)
    vali_list = os.listdir(vali_path)

    ## Parametros para el almacenamiento de las imagenes
    width = 200
    heigth = 200
    # Entrenamiento
    labels_train = []
    imag_train = []
    data_train = []
    cont_train = 0
    # Validacion
    labels_val = []
    imag_val = []
    data_val = []
    cont_val = 0

    ## Extraer fotos y etiquetarlas

    for folder_name in train_list:
        name_path = train_path + '/' + folder_name
        folder = os.listdir(name_path)
        for file in folder:
            labels_train.append(cont_train)
            img = cv.imread(name_path + '/' + file, 0)
            img = cv.resize(img, (width, heigth), interpolation=cv.INTER_CUBIC)
            img = img.reshape(width, heigth, 1)
            data_train.append([img, cont_train])
            imag_train.append(img)
        cont_train += 1

    for folder_name in vali_list:
        name_path = vali_path + '/' + folder_name
        folder = os.listdir(name_path)
        for file in folder:
            labels_val.append(cont_val)
            img = cv.imread(name_path + '/' + file, 0)
            img = cv.resize(img, (width, heigth), interpolation=cv.INTER_CUBIC)
            img = img.reshape(width, heigth, 1)
            data_val.append([img, cont_val])
            imag_val.append(img)
        cont_val += 1

    ## Normalizar imagenes
    imag_train = np.array(imag_train).astype(float) / 255
    print(imag_train.shape)
    imag_val = np.array(imag_val).astype(float) / 255
    print(imag_val.shape)
    # Pasar las listas a arrays
    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)

    ## Dar realismo a las imagenes
    # Modificar las imagenes aleatoriamente

    image_random_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=10,
        zoom_range=[0.2, 1],
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_train_gen = image_random_gen.flow(imag_train, labels_train, batch_size=32)

    ## Creamos el modelo de la red convolucional con Drop Out
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
        # capa convolucional con 32 Kernel
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # capa convolucional con 64 Kernel
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # capa convolucional con 128 Kernel
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ## Compilamos el modelo y añadimos una funcion de perdida y el optimizador
    cnn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ## Entrenar la red neuronal
    # tensorboard --logdir="C:/RedNeuronal/board"

    board = TensorBoard(log_dir='C:/RedNeuronal/board')
    cnn_model.fit(
        image_train_gen,
        batch_size=32,
        validation_data=(imag_val, labels_val),
        epochs=epoch,
        callbacks=[board],
        steps_per_epoch=int(np.ceil(len(imag_train) / float(32))),
        validation_steps=int(np.ceil(len(imag_val) / float(32)))
    )

    cnn_model.save('clasificador1.h5')
    print('Red Nueronal Terminada')


def ball_command():
    global detector_ball
    # detect, use 1,7 and 20
    path_ball = os.path.dirname(__file__) + '/cascade_balls.xml'
    detector_ball = cv.CascadeClassifier(path_ball)
    print(detector_ball)
    pass

def animals_command():
    ## path de nuestro modelo
    cnn_model = os.path.dirname(__file__) + '/CatsDogs.h5'

    ## Leer la red neuronal
    detector_2 = tf.keras.models.load_model(cnn_model)
    print(detector_2)


def human_command():
    path_face = os.path.dirname(__file__) + '/haarcascade_frontalface_default.xml'
    path_body = os.path.dirname(__file__) + '/haarcascade_fullbody.xml'
    detector_face = cv.CascadeClassifier(path_face)
    detector_body = cv.CascadeClassifier(path_body)
    detector_2 = False
    detector_ball = False
    print(detector_body, detector_face)


def use():
    path = object_path_text.get()
    print(path)
    cnn_model = path
    detector_2 = tf.keras.models.load_model(cnn_model)
    print(detector_2)


def useObjectWindow():
    # Toplevel object which will
    newWindow = tk.Toplevel(master)
    # sets the title of the
    newWindow.title("New Window")
    # sets the geometry of toplevel
    newWindow.geometry("400x400")

    input_1_label = tk.Label(newWindow,text="Object detection path").place(x=45, y=60)
    submit_button = tk.Button(newWindow,text="Submit", command=use).place(x=45,y=110)
    text_1 = tk.StringVar()
    global object_path_text
    input_1 = tk.Entry(newWindow, textvariable=text_1, width=50).place(x=45,y=85)
    object_path_text =text_1
    newWindow.destroy()


def trainObjectWindow():
    # Toplevel object which will
    newWindow = tk.Toplevel(master)
    # sets the title of the
    newWindow.title("New Window")
    # sets the geometry of toplevel
    newWindow.geometry("400x400")

    input_1_label = tk.Label(newWindow,text="Train path").place(x=45, y=60)
    input_2_label = tk.Label(newWindow,text="Validation path").place(x=40,y=100)
    input_3_label = tk.Label(newWindow, text="Epocas").place(x=40, y=140)
    submit_button = tk.Button(newWindow,text="Submit", command=train).place(x=40,y=180)
    text_1 = tk.StringVar()
    text_2 = tk.StringVar()
    text_3 = tk.StringVar()
    global input_1_text
    input_1 = tk.Entry(newWindow, textvariable=text_1, width=30).place(x=130,y=60)
    input_1_text =text_1
    global input_2_text
    input_2 = tk.Entry(newWindow, textvariable=text_2, width=30).place(x=130,y=100)
    input_2_text = text_2
    global input_3_text
    input_3 = tk.Entry(newWindow, textvariable=text_3, width=30).place(x=130,y=140)
    input_3_text = text_3
    newWindow.destroy()

my_menu = Menu(master)
master.config(menu=my_menu)
red_menu = Menu(my_menu)
my_menu.add_cascade(menu=red_menu, label='Redes Neuronales')
red_menu.add_command(label='Human detection', command=human_command)
red_menu.add_separator()
red_menu.add_command(label="Ball detection", command=ball_command)
red_menu.add_separator()
red_menu.add_command(label="Cats & dogs detection", command=animals_command)
red_menu.add_separator()
red_menu.add_command(label="Train a new object recognition", command=trainObjectWindow)
red_menu.add_separator()
red_menu.add_command(label="Use the new object recognition", command=useObjectWindow)


def takePictureButtonClicked():
    print ("Take picture")
    client.publish("dashBoard/cameraService/takePicture")

takePictureButton = tk.Button(takePictureFrame, text="Take Picture", width=50, bg='red', fg="white", command=takePictureButtonClicked)
takePictureButton.grid(column=0, row=0, pady = 20, padx = 20)

img = Image.open("image1.jpg")
img = img.resize((350, 350), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = tk.Label(takePictureFrame, image=img, borderwidth=2, relief="raised")
panel.image = img
panel.grid(column=0, row=1, columnspan=3, rowspan = 3)




videoStreamFrame = tk.Frame(cameraControlFrame)
videoStreamFrame.pack()

videoStream = False;

def videoStreamButtonClicked():
    global videoStream
    global client
    if not videoStream:
        videoStreamButton['text'] = "Stop video stream"
        videoStreamButton['bg'] = "green"
        videoStream = True
        print ('Start video stream')
        client.publish("dashBoard/cameraService/startVideoStream")

    else:
        videoStreamButton['text'] = "Start video stream on a separaded window"
        videoStreamButton['bg'] = "red"
        videoStream = False
        print ('Stop video stream')
        client.publish("dashBoard/cameraService/stopVideoStream")

        cv.destroyWindow("Stream")



videoStreamButton = tk.Button(videoStreamFrame, text="Start video stream \n on a separaded window", width=50, height = 25, bg='red', fg="white",
                              command=videoStreamButtonClicked)
myFont = font.Font(size=12)
videoStreamButton['font'] = myFont
videoStreamButton.grid(column=0, row=0, pady=20, padx=20, )



master.mainloop()