import json
import dronekit

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
print(points_list)
first_point = points_list[0]
print(first_point)
first_point = points_list[0]
originPoint = dronekit.LocationGlobalRelative(first_point[0], first_point[1], 20)
position = str(first_point[0]) + '*' + str(first_point[0]) + '*0'
print(position, 'First')
for wp in points_list[1:]:
    print ('Siguiente punto del flight plan')
    print (wp)
    if len(wp) == 3:
        print(wp, 'photo')
