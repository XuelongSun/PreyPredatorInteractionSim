from enum import Flag
import time
import socket
from threading import Thread
import pickle

import numpy as np
import trimesh as tr
import pyrender as pr

class DataReceiver(Thread):
    def __init__(self, socket):
        super().__init__()
        self.data = None
        self.socket = socket
        self.socket_is_connected = False
    
    def recvive_all(self, count):
        buffer = b''
        while count:
            try:
                buf = self.socket.recv(count)
                if not buf:
                    return None
                buffer += buf
                count -= len(buf)
            except socket.error as e:
                return False
        return buffer
    
    def run(self):
        while self.socket_is_connected:
            try:
                # first get the length of the data
                data_length = self.recvive_all(16)
                if data_length:
                    # get data
                    self.data = self.recvive_all(int(data_length))
                    if self.data:
                        pass
                    else:
                        break
                else:
                    break
            except socket.error as e:
                return False

class DataServer:
    '''
        use socket to send key data to GUI for analysis
    '''
    def __init__(self, data, ip='127.0.0.1', port=6666):
        self.ip = ip
        self.port = port
        self.data = data
        self.stop = False
        
    def data_handle(self, new_socket, addr):
        try:
            while (len(self.data)>0) and (not self.stop):
                send_data = pickle.dumps(self.data)
                new_socket.sendall(str.encode(str(len(send_data)).ljust(16)))
                new_socket.sendall(send_data)
                time.sleep(0.1)
        except Exception as ret:
            print(str(addr) + " error, disconnected..: " + str(ret))
        
    def run(self):
        try:
            main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            main_socket.bind((self.ip, self.port)) 
            main_socket.listen(128)  
            print("sever started...")
            while not self.stop:
                new_socket, addr = main_socket.accept()
                Thread(target=self.data_handle, args=(new_socket, addr)).start()
        except Exception as ret:
            print("server error: " + str(ret))

def world2image_coordinates_transfer(p, boundary):
    c = int(np.clip(p[0], boundary[0], boundary[1]-1) - boundary[0])
    r = int(boundary[3] - np.clip(p[1], boundary[2]+1, boundary[3]))
    return [r, c]

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calculate_pose_matrix(rotation, translation):
    '''
    create camera pose matrix in homogeneous format.add()
    
    Parameters
    ---
    rotation - dict: {'x':r, 'y': phi, 'z': theta}
    translation - array: [x, y, z]
    '''
    M = np.identity(4)
    
    # rotation
    rotate_matrix = np.identity(3)
    if 'x' in rotation.keys():
        angle = rotation['x']
        if not angle == 0:
            Rx = np.zeros(shape=(3, 3))
            Rx[0, 0] = 1
            Rx[1, 1] = np.cos(angle)
            Rx[1, 2] = -np.sin(angle)
            Rx[2, 1] = np.sin(angle)
            Rx[2, 2] = np.cos(angle)
            rotate_matrix = np.matmul(rotate_matrix, Rx)
    if 'y' in rotation.keys():
        angle = rotation['y']
        if not angle == 0:
            Ry = np.zeros(shape=(3, 3))
            Ry[0, 0] = np.cos(angle)
            Ry[0, 2] = -np.sin(angle)
            Ry[2, 0] = np.sin(angle)
            Ry[2, 2] = np.cos(angle)
            Ry[1, 1] = 1
            rotate_matrix = np.matmul(rotate_matrix, Ry)
    if 'z' in rotation.keys():
        angle = rotation['z']
        if not angle == 0:
            Rz = np.zeros(shape=(3, 3))
            Rz[0, 0] = np.cos(angle)
            Rz[0, 1] = -np.sin(angle)
            Rz[1, 0] = np.sin(angle)
            Rz[1, 1] = np.cos(angle)
            Rz[2, 2] = 1
            rotate_matrix = np.matmul(rotate_matrix, Rz)
    M[:3, :3] = rotate_matrix
    # translation
    M[:3, 3] = translation
    
    return M

def create_cylinder_robot_node(radius, height, color=(135, 206, 235)):
    bar = tr.creation.box(extents=(0.5, radius*0.9, 0.1))
    bar.visual.face_colors = (229, 240, 16)
    pose = calculate_pose_matrix({},[0.0, radius/2, height/2])
    bar_node = pr.Node(mesh=pr.Mesh.from_trimesh(bar, smooth=False),
                       matrix=pose)
    camera = pr.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.2)
    pose = calculate_pose_matrix({'x':np.pi/2,'y':0},[0,radius,0])
    camera_node = pr.Node(camera=camera, matrix=pose)
    
    c = tr.creation.cylinder(radius=radius, height=height)
    c.visual.face_colors = color
    c_node = pr.Node(mesh=pr.Mesh.from_trimesh(c, smooth=False),
                     children=[bar_node, camera_node])
    
    return c_node

def color_rgb2hsv(color):
    '''
    convert RGB color to HSV
    
    HSV is in the range as open-cv: i.e.:
    H : 0 ~ 255
    S : 0 ~ 255
    V : 0 ~ 255
    color: rgb(0-255, 0-255, 0-255)
    '''
    c_max = max(color)
    c_min = min(color)
    
    if c_max == c_min:
        h = 0
    elif color.index(c_max) == 0 and color[1] >= color[2]:
        h = 60 * (color[1] - color[2])/(c_max - c_min)
    elif color.index(c_max) == 0 and color[1] < color[2]:
        h = 60 * (color[1] - color[2])/(c_max - c_min) + 360
    elif color.index(c_max) == 1:
        h = 60 * (color[2] - color[0])/(c_max - c_min) + 120
    elif color.index(c_max) == 2:
        h = 60 * (color[0] - color[1])/(c_max - c_min) + 240
    
    s = 0 if c_max == 0 else (c_max - c_min)/c_max
    
    v = c_max
    
    return (int(h/2), int(s*255), int(v))