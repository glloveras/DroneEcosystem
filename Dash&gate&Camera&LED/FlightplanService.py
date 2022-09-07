import math
import ast
import paho.mqtt.client as mqtt
import tkinter
import json
import tkinter.messagebox
from Tkintermap import TkinterMap
import threading


def openFlightPlan():
    print('Opening Flightplan map')
    flightplan = Flightplan()
    flightplan.start()


def on_message(client, userdata, message):
    splited = message.topic.split('/')
    origin = splited[0]
    destination = splited[1]
    command = splited[2]

    if command == 'openFlightplan':
        client.subscribe('+/flightplanService/#')
        w = threading.Thread(target=openFlightPlan)
        w.start()


global_broker_address =  "127.0.0.1"
global_broker_port = 1884
LEDSequenceOn = False
client = mqtt.Client("Flightplan service")
client.on_message = on_message
client.connect(global_broker_address, global_broker_port)
client.loop_start()
client.subscribe('+/flightplanService/openFlightplan')


def calc_pixels(tupla_1, tupla_2):
    pixels = math.sqrt(math.pow((tupla_1[0]-tupla_2[0]), 2) + math.pow((tupla_1[1]-tupla_2[1]), 2))
    return pixels


class Flightplan(tkinter.Tk):

    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)

        self.geometry(f"{1200}x{800}")
        self.title("Flightplan Map")
        self.protocol("WM_DELETE_WINDOW", self.closing)
        self.bind("<Return>", self.search)

        # ============ create the general grid ============

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # ============ search configuration ============

        self.search_bar = tkinter.Entry(self, width=50)
        self.search_bar.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")
        self.search_bar.focus()

        self.button_1 = tkinter.Button(master=self, text="Search", command=self.search, width=15)
        self.button_1.grid(pady=10, padx=20, row=2, column=1)

        self.button_2 = tkinter.Button(master=self, text="Clear", command=self.clear, width=15)
        self.button_2.grid(pady=10, padx=20, row=2, column=2)

        # ============ map ============

        self.map_widget = TkinterMap(self, width=800, height=600, corner_radius=0, max_zoom=19)
        self.map_widget.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=20, pady=20)
        self.map_widget.set_address("-35.363 149.165")
        self.map_widget.add_right_click_menu_command(label="Add point", command=self.set_point, pass_coords=True, pass_pixels=True)

        # ============ coordinates menu ============

        self.listbox_button_frame = tkinter.Frame(master=self)
        self.listbox_button_frame.grid(row=0, column=1, sticky="nsew", columnspan=3)
        self.listbox_button_frame.grid_columnconfigure(0, weight=1)
        self.listbox_button_frame.grid_rowconfigure(4, weight=1)

        self.clear_marker_button = tkinter.Button(
            master=self.listbox_button_frame, width=20, text="Clear points list", command=self.clear_points_list
        )
        self.clear_marker_button.grid(row=0, column=0, pady=10, padx=10)

        self.send_points_button = tkinter.Button(
            master=self.listbox_button_frame, width=20, text="Send flight plan", command=self.send_points
        )
        self.send_points_button.grid(row=3, column=0, pady=10, padx=10)

        self.point_list_box = tkinter.Listbox(master=self.listbox_button_frame, height=20, selectmode="extended")
        self.point_list_box.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

        photo_button_add = tkinter.Button(master=self.listbox_button_frame, width=20, text="Add point to Take photo list", command=self.selected_items)
        photo_button_add.grid(row=5, column=0, sticky="nsew", padx=10, pady=10)
        photo_button_rem = tkinter.Button(master=self.listbox_button_frame, width=20, text="Remove point from Take photo list", command=self.delete_items)
        photo_button_rem.grid(row=6, column=0, sticky="nsew", padx=10, pady=10)

        self.search_address = None
        self.searching = False
        self.points_list = []
        self.points_path = None
        self.last_tram = None
        self.ptp_path = None
        self.previous_path = []
        self.line_text = None
        self.pixels_list = []
        self.factor_p_to_m = 0.28
        self.new_point = None
        self.first_point = False
        self.last_point = False
        self.photo_coord_set = set()

    def search(self, event=None):

        if not self.searching:
            self.searching = True
            self.address = self.search_bar.get()
            self.search_address = self.map_widget.set_address(self.address, marker=True)
            self.map_widget.set_zoom(zoom=19)
            if self.search_address is False:
                self.search_address = None
            self.searching = False

    def set_point(self, coords, mouse_pixels):

        print("Add point:", coords)
        print('pixel', mouse_pixels)
        positions_list = []

        if self.first_point is False:
            self.new_point = self.map_widget.set_marker(coords[0], coords[1], text="")
            self.point_list_box.insert(tkinter.END, f"{coords}; {'First'}")
            self.point_list_box.see(tkinter.END)
            self.points_list.append(self.new_point)
            self.pixels_list.append(mouse_pixels)
            self.map_widget.delete(self.last_point)
            self.last_point = False
            self.first_point = True
        else:
            if self.last_tram:
                self.map_widget.delete(self.last_tram)
            if self.last_point:
                self.map_widget.delete(self.last_point)
                self.point_list_box.delete(tkinter.END, tkinter.END)
                self.points_list.pop(-1)
            self.pixels_list.append(mouse_pixels)
            pixels = calc_pixels(self.pixels_list[-1], self.pixels_list[-2])
            dist = pixels * self.factor_p_to_m
            self.new_point = self.map_widget.set_marker(coords[0], coords[1], text="{} m".format(round(dist, 2)))
            self.point_list_box.insert(tkinter.END, f"{coords}; {round(dist, 2)}")
            self.point_list_box.see(tkinter.END)
            self.points_list.append(self.new_point)
            self.previous_path.append(self.ptp_path)

            for point in self.points_list:
                positions_list.append(point.position)

            self.ptp_path = self.map_widget.set_path(positions_list)

            pixels = calc_pixels(self.pixels_list[0], self.pixels_list[-1])
            dist = pixels * self.factor_p_to_m

            first = positions_list[0]
            last = positions_list[-1]
            last_tram = [first, last]
            self.last_tram = self.map_widget.set_path(last_tram)
            self.last_point = self.map_widget.set_marker(first[0], first[1], text="{} m".format(round(dist, 2)))
            self.points_list.append(self.last_point)
            self.point_list_box.insert(tkinter.END, f"{first}; {'Last'}")

    def clear_points_list(self):

        for point in self.points_list:
            self.map_widget.delete(point)
        for path in self.previous_path:
            self.map_widget.delete(path)

        self.point_list_box.delete(0, tkinter.END)
        self.points_list.clear()
        self.pixels_list.clear()
        self.map_widget.delete(self.ptp_path)
        self.map_widget.delete(self.last_tram)
        self.map_widget.delete(self.last_point)
        self.photo_coord_set.clear()
        self.first_point = False
        self.last_point = False

    def send_points(self):
        position_list = []
        position_list_photo = []
        new_list_points = []

        for point in self.points_list:
            position_list.append(point.position)

        for coord in self.photo_coord_set:
            position_list_photo.append(coord)

        for x in range(len(position_list)):
            if position_list[x] in position_list_photo:
                tuple_position = position_list[x]
                photo_tuple = tuple_position + (1,)
                new_list_points.append(photo_tuple)
            else:
                new_list_points.append(position_list[x])

        position_string = json.dumps(new_list_points)

        # Enviar al autoplioto
        client.publish('flightplanService/autopilotService/flightplanpoints', position_string)
        print('flightplanService/autopilotService/flightplanpoints', position_string)

    def selected_items(self):
        for i in self.point_list_box.curselection():
            info = self.point_list_box.get(i)
            splited = info.split(";")
            coord_tuple = ast.literal_eval(splited[0])
            self.photo_coord_set.add(coord_tuple)

        print(self.photo_coord_set)

    def delete_items(self):
        for i in self.point_list_box.curselection():
            info = self.point_list_box.get(i)
            splited = info.split(";")
            coord_tuple = ast.literal_eval(splited[0])
            self.photo_coord_set.discard(coord_tuple)

        print(self.photo_coord_set)

    def clear(self):
        self.search_bar.delete(0, last=tkinter.END)
        self.map_widget.delete(self.search_address)

    def closing(self):
        self.destroy()
        exit()

    def start(self):
        self.mainloop()