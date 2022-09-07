import json
import os.path

"""
mi_set = {(-35.362878311001616, 149.16443342018692), (-35.36255239529314, 149.16463726807206), (-35.36281487784604, 149.16486525583832), (-35.36260270451522, 149.16527831602662)}
position = []
for coord in mi_set:
    position.append(coord)
print(position)
"""
photo_list_points = [(-35.362545833218384, 149.16508251476853), (-35.36233803390858, 149.16436636496155), (-35.362797379035726, 149.16492962885468)]
list_points = [(-35.362633327504625, 149.16448974657624), (-35.362797379035726, 149.16492962885468), (-35.362545833218384, 149.16508251476853), (-35.36238396853889, 149.1647552852687), (-35.36233803390858, 149.16436636496155), (-35.362633327504625, 149.16448974657624)]
#newlist = [x for x in mylist if x%2 == 0]
#new_list_points = [point for point in list_points if (point in photo_list_points)]
#print(new_list_points)
new_list_points = []
for x in range(len(list_points)):
    if list_points[x] in photo_list_points:
        tuple = list_points[x]
        photo_tuple = tuple + (1,)
        new_list_points.append(photo_tuple)
    else:
        new_list_points.append(list_points[x])

string = json.dumps(new_list_points)
points_list = json.loads(string)
fist_point = points_list[0]

print(os.path.dirname(__file__))

def global_var():
    global detector
    if detector:
        print(detector)

detector = True
global_var()

"""

        if command == 'picture':
            print('show picture')
            img = base64.b64decode(message.payload)
            # converting into numpy array from buffer
            npimg = np.frombuffer(img, dtype=np.uint8)
            # Decode to Original Frame
            cv2image = cv.imdecode(npimg, 1)
            if v5.get() == "1":
                print("Human")
                # haarcascade_fullbody.xml -> for human fullbody detection
                # haarcascade_frontalface_default.xml -> for face detection
                bodyDetect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
                # change to one color
                gray = cv.cvtColor(cv2image, cv.COLOR_BGR2GRAY)
                # detect face, if detected will save x,y,w,h
                body = bodyDetect.detectMultiScale(gray, 1.2, 5)
                # Drawing the rectangle
                for (x, y, w, h) in body:
                    cv.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if v5.get() == "2":
                print("Cars")
                carDetect = cv.CascadeClassifier('cascade.xml')
                gray = cv.cvtColor(cv2image, cv.COLOR_BGR2GRAY)
                # detect face, if detected will save x,y,w,h
                car = carDetect.detectMultiScale(gray, 1.4, 50)
                # Drawing the rectangle
                for (x, y, w, h) in car:
                    cv.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if v5.get() == "3":
                print("Dogs & Cats")
            dim = (300, 300)
"""