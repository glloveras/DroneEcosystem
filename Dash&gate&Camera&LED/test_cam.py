import tensorflow as tf
import numpy as np
import cv2 as cv
from keras_preprocessing.image import img_to_array

"""
## path de nuestro modelo
#cnn_model = 'C:/RedNeuronal/CatsDogs.h5'

## Leer la red neuronal
#cnn = tf.keras.models.load_model(cnn_model)

## Empezar la videocaptura
video = cv.VideoCapture(0)
detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    _, frame = video.read()
    # change to one color
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detect, if detected will save x,y,w,h
    object = detect.detectMultiScale(gray, 1.2, 5)
    #object = detect.detectMultiScale(gray, 1.35, 20)
    # Drawing the rectangle
    for (x, y, w, h) in object:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #gray = cv.resize(gray, (200, 200), interpolation=cv.INTER_CUBIC)
    #gray = np.array(gray).astype(float) / 255
    #img = img_to_array(gray)
    #img = np.expand_dims(img, axis=0)

    #predict = cnn.predict(img)
    #predict = predict[0][0]
    #print(predict)
    #if predict > 0.93:
    #    cv.putText(frame, "Cat detected", (200, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))

    cv.imshow('CAM', frame)
    a = cv.waitKey(1)
    if a > 0:
        break

cv.destroyAllWindows()
video.release()
"""
"""

def onMouseAction(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print("(B,G,R)", img[y, x])
        # obtain the image in hsv color space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        print("(H,S,V)", hsv[y, x])

img = cv.imread('C:/RedNeuronal/colores.jpg')

cv.imshow("image", img)
cv.setMouseCallback("image", onMouseAction)
cv.waitKey(0)
"""

"""

img = cv.imread('C:/RedNeuronal/exper.jpg')

print('Original Dimensions : ', img.shape)

scale_percent = 30  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv.imshow("Resized image", resized)

copy = resized.copy()
image = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)


# lower boundary RED color range values
r_lower_1 = np.array([0, 120, 110])
r_upper_1 = np.array([10, 255, 255])
# upper boundary RED color range values
r_lower_2 = np.array([170, 170, 75])
r_upper_2 = np.array([180, 255, 255])

r_lower_mask = cv.inRange(image, r_lower_1, r_upper_1)
r_upper_mask = cv.inRange(image, r_lower_2, r_upper_2)
r_full_mask = r_lower_mask + r_upper_mask
# r_result = cv.bitwise_and(copy, copy, mask=r_full_mask)

kernel = np.ones((1, 3), np.uint8)
print(kernel)
# this is opening
r_opening = cv.morphologyEx(r_full_mask, cv.MORPH_OPEN, kernel)
# this is closing
r_clean_mask = cv.morphologyEx(r_opening, cv.MORPH_CLOSE, kernel)

cv.imshow('Mask', r_full_mask)
cv.imshow("Clened Mask", r_clean_mask)
# cv.imshow('Result', r_result)

# Find edges
#edged = cv.Canny(r_clean_mask, 2, 200)
#edged_details = cv.Canny(r_full_mask, 2, 200)
contours, hierarchy = cv.findContours(r_clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#contours_details, hierarchy_Details = cv.findContours(edged_details, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(resized, contours, -1, (0, 255, 0), 2)
#cv.drawContours(resized, contours_details, -1, (255, 0, 0), 1)
cv.imshow("Original with contours", resized)

cv.waitKey(0)
cv.destroyAllWindows()

########################################################################################################################
"""
"""
params = cv.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 1
params.maxArea = 10000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter bt Color
params.filterByColor = True
params.blobColor = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0

# Create a detector with the parameters
ver = (cv.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)
"""

########################################################################################################################

path_list = ['C:/RedNeuronal/tball_court.jpg', 'C:/RedNeuronal/tballs.jpg']
image_list = [cv.imread(path) for path in path_list]

for img in image_list:
    print('Original Dimensions : ', img.shape)

    scale_percent = 40  # percent of original size
    if img.shape[0] < 600:
        scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv.imshow("Resized image", resized)

    copy = resized.copy()
    image = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)

    # Range for  YELLOW
    yellow_lower_1 = np.array([20, 10, 10])
    yellow_upper_1 = np.array([50, 255, 255])
    y_mask = cv.inRange(image, yellow_lower_1, yellow_upper_1)
    y_result = cv.bitwise_and(copy, copy, mask=y_mask)

    kernel = np.ones((1, 3), np.uint8)
    print(kernel)
    # this is opening
    y_opening = cv.morphologyEx(y_mask, cv.MORPH_OPEN, kernel)
    # this is closing
    y_clean_mask = cv.morphologyEx(y_opening, cv.MORPH_CLOSE, kernel)

    # Aplicar suavizado Gaussiano
    # y_clean_mask = cv.GaussianBlur(y_mask, (5, 5), 0)

    cv.imshow('Mask', y_mask)
    cv.imshow("Cleaned Mask", y_clean_mask)
    cv.imshow('Result', y_result)

    # Find edges
    # edged = cv.Canny(y_clean_mask, 30, 150)
    # edged_details = cv.Canny(y_mask, 30, 150)
    contours, hierarchy = cv.findContours(y_clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours_details, hierarchy_Details = cv.findContours(y_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(resized, contours, -1, (0, 255, 0), 2)
    # cv.drawContours(resized, contours_details, -1, (255, 0, 0), 1)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 300:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            print(len(approx))
            if len(approx) in (8, 9, 10, 11):
                cv.drawContours(resized, contour, -1, (255, 0, 0), 2)

    # keypoints = detector.detect(y_clean_mask)
    # print(len(keypoints))
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv.drawKeypoints(copy, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    # cv.imshow("Keypoints", im_with_keypoints)

    cv.imshow("Original with contours", resized)
    cv.waitKey(0)

cv.destroyAllWindows()
