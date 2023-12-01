from collections import deque

import numpy as np
import pyrender as pr
import trimesh as tm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from utils import world2image_coordinates_transfer, distance, color_rgb2hsv
from utils import create_cylinder_robot_node, calculate_pose_matrix

# 3d view
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 150

scene3d = pr.Scene(ambient_light=1.0, bg_color=(0,0,0))
render3d = pr.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)

class OdorSensor:
    def __init__(self) -> None:
        self.radius = 1
        self.value = 0
    
    def get_value(self, p, phi_field, boundary):
        p_ = world2image_coordinates_transfer(p, boundary)
        self.value = phi_field[p_[0], p_[1]]
        return self.value

class Agent:
    def __init__(self, uuid, init_heading, init_position,
                 raduis=10, height=5, color=(135, 206, 235)):
        self.id = uuid
        self.cluster_id = None

        # motion
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.position = np.array(init_position, np.float32)
        self.heading = init_heading
        
        self.collision = False
        self.collision_theta = 0
        self.collision_counter = 0
        
        # scale
        self.size = raduis
        self.collision_distance = self.size
        # visual 3D object
        self.node3d = create_cylinder_robot_node(raduis, height, color)
        scene3d.add_node(self.node3d)
        pose = calculate_pose_matrix({'z':self.heading-np.pi/2},
                                     [self.position[0],
                                      self.position[1], 0])
        scene3d.set_pose(self.node3d, pose=pose)
        self.body_color_rgb = color
        self.body_color_hsv = color_rgb2hsv(color)
        
        self.time = 0
        
        self.last_cluster_prey_num = 0

    def update_motion(self, boundary, obstacles):
        # if still in collision
        if self.collision_counter > 0:
            self.linear_velocity = 2
            self.angular_velocity = 0
            self.collision_counter -= 1
        else:
            # if at  the arena boundary
            if (self.position[0] < boundary[0] + self.size) or \
                (self.position[0] > boundary[1] - self.size) or \
                    (self.position[1] < boundary[2] + self.size) or \
                        (self.position[1] > boundary[3] - self.size):
                            # sharp turn
                            self.collision_counter = 20
                            self.linear_velocity = 1.2
                            self.angular_velocity = np.random.uniform(np.pi/2, np.pi)
            elif self.collision_obstacles(obstacles):
                self.angular_velocity = np.random.vonmises(3*np.pi/4, 100.0, 1)[0]
            
            if self.collision:
                # completely inelastic collision
                self.linear_velocity *= np.sin(self.heading - self.collision_theta)
                self.angular_velocity -= self.collision_theta

        self.heading += self.angular_velocity
        # scale to -pi~pi
        self.heading = (self.heading + np.pi) % (np.pi*2) - np.pi
        self.position += self.linear_velocity * np.array([np.cos(self.heading),
                                                        np.sin(self.heading)])
        # print('position:',self.position)
        # update visual object in 3d scene
        pose = calculate_pose_matrix({'z':self.heading-np.pi/2},
                                    [self.position[0], self.position[1], 0])
        scene3d.set_pose(self.node3d, pose=pose)
        
    
    def get_camera_data(self):
        scene3d.main_camera_node = self.node3d.children[1]
        c, d = render3d.render(scene3d)
        return c

    def collision_obstacles(self, obstacles):
        if obstacles is not None:
            for obstacle in obstacles:
                if obstacle['shape'] == 'circle':
                    if distance(self.position, obstacle['position']) <= \
                        obstacle['radius'] + self.size + self.collision_distance:
                            return True
                elif obstacle['shape'] == 'rectangle':
                    margin_x = self.size + obstacle['width']/2 + self.collision_distance
                    margin_y = self.size + obstacle['height']/2 + self.collision_distance
                    if abs(self.position[0] - obstacle['center'][0]) <= margin_x and \
                        abs(self.position[1] - obstacle['center'][1]) <= margin_y:
                            return True
            return False
        else:
            return False

class Prey(Agent):
    def __init__(self, uuid, init_heading, init_position):
        super().__init__(uuid, init_heading, init_position)
        
        self.odor_sensors = [OdorSensor(), OdorSensor()]
        # self.energy = 20
        self.energy = 2
        self.state = "wandering"
        
        # evolving parameters
        # self.f_gather = np.random.rand(1)[0] * 10 + 5
        # self.f_avoid = np.random.rand(1)[0] * 0.5 + 0.1
        
        self.f_gather = np.random.uniform(0.02, 0.32)
        self.f_avoid = np.random.uniform(0.01,10.01)
        
        self.odor_sensors_positions = np.array([[0, self.size/2], [0, -self.size/2]])
        self.view = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], np.uint8)
        self.view_processed = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], np.uint8)
        
        # visual based control
        self.hsv_l = (self.body_color_hsv[0] - 5, 50, 50)
        self.hsv_h = (self.body_color_hsv[0] + 5, 255, 255)
        self.thr_start_pixel_num = 0.05
        self.thr_stop_pixel_num = 0.4
        self.max_pixel_num = 0.7
        self.f_size = 0
        
        # olfactory based control
        self.odor_sensor_l_values = deque([0,0,0,0],maxlen=4)
        self.odor_sensor_r_values = deque([0,0,0,0],maxlen=4)
        
    def get_odor_sensor_position(self):
        positions = []
        r_matrix = np.array([[np.cos(self.heading), np.sin(self.heading)],
                             [-np.sin(self.heading), np.cos(self.heading)]])
        for p in self.odor_sensors_positions:
            rotated = np.matmul(p, r_matrix)
            positions.append(self.position + rotated)
        return positions
    
    def update(self, dt, phero_f, boundary):
        if (self.energy <= 0) and (self.state != 'death'):
            self.state = "death"
            # remove from the 3d scene
            scene3d.remove_node(self.node3d)
            return
        elif(self.energy>=30):
            self.energy=30
        
        if self.state == 'death':
            return
        
        # get predator's alarm pheromone
        s_p = self.get_odor_sensor_position()
        o_value_l = self.odor_sensors[0].get_value(
            s_p[0], phero_f, boundary
        )[0]
        self.odor_sensor_l_values.append(o_value_l)
        o_value_r = self.odor_sensors[1].get_value(
            s_p[1], phero_f, boundary
        )[0]
        self.odor_sensor_r_values.append(o_value_r)
        # get camera data (image view)
        if self.time % 100 < dt:
            self.view = self.get_camera_data()
        
        # state update depending on sensory inputs
        if (o_value_l > self.f_avoid) or (o_value_r > self.f_avoid):
            self.state = 'avoiding'
        else:
            # find if there is interesting color in the view
            hsv = cv2.cvtColor(self.view, cv2.COLOR_RGB2HSV)
            img_f = cv2.inRange(hsv, self.hsv_l, self.hsv_h)
            size = img_f.sum()/7650000
            # tim2 = self.view.shape
            # print('img_f.sum = ', tim2)
            # print('size:', size)
            if size >= self.max_pixel_num:
                self.state = 'stop'
            elif size >= self.thr_stop_pixel_num:
                self.state = 'stop'
            elif size >= self.f_gather:
                self.state = 'gathering'
            else:
                self.state = 'wandering'

            self.view_processed = img_f.copy()
            self.f_size = size

        # motion control depend on state
        if self.state == 'avoiding':
            # f = np.max([self.energy * 0.2, 4])
            f = np.max([0.2, np.min([self.energy * 0.2, 4])])
            self.energy -= f * dt / 1000
            # self.linear_velocity = np.min([f, 2.4])
            self.linear_velocity = f
            
            odor_old = self.odor_sensor_l_values[-2] + self.odor_sensor_r_values[-2]
            odor_now = self.odor_sensor_l_values[-1] + self.odor_sensor_r_values[-1]
            if odor_now > odor_old:
                self.angular_velocity = np.pi/4*3
            else:
                self.angular_velocity = 0
            # self.angular_velocity = (o_value_l - o_value_r)*10 % (np.pi*3/4)
                
        elif self.state == 'gathering':
            self.energy -= 0.2 * dt / 1000
            self.linear_velocity = 1.2
            # depend on the visual input
            x = np.mean(np.where(img_f > 1)[1])
            self.angular_velocity = (IMAGE_WIDTH/2 - x) / IMAGE_WIDTH * (np.pi/4)
        elif self.state == 'stop':
            self.energy += 2 * dt / 1000
            self.linear_velocity = 0 
            self.angular_velocity = 0
        elif self.state == 'wandering':
            self.energy -= 0.2 * dt / 1000
            self.linear_velocity = 1
            self.angular_velocity = np.random.vonmises(0.0, 100.0, 1)[0]

        self.update_motion(boundary, None)
        
        self.time += dt

class Predator(Agent):
    def __init__(self, uuid, init_heading, init_position, speed=4):
        super().__init__(uuid, init_heading, init_position, color=(200, 0, 20))
        self.phero_radius = 40
        self.linear_velocity = speed
        self.goal_defined = False
        self.energy = 16
        self.state = 'hunting'
        
    def update(self, boundary, cluster, num_prey):
        self.energy -= self.energy*0.0001
        if 18 <= self.energy:
            self.state = 'stopping'
            # no motion
            return
        else:
            self.state = 'hunting'
            # found cluster with highest density
            sorted_c = sorted(cluster.items(), key=lambda v:len(v[1]))
            _, v1 = sorted_c[-1]
            v2 = [] if len(sorted_c) <=1 else sorted_c[-2][1]

            # if cluster size >= 5
            if len(v1) >= 5:
                if len(v1) >= num_prey - 2:
                    i = np.random.randint(0, len(v1))
                    px = v1[i].position[0]
                    py = v1[i].position[1]
                    goal_dir = np.arctan2(self.position[1]-py, self.position[0]-px)
                else:
                    if len(v1) >= self.last_cluster_prey_num+2:
                        px = np.array([a.position[0] for a in v1]).mean()
                        py = np.array([a.position[1] for a in v1]).mean()
                        goal_dir = np.arctan2(self.position[1]-py, self.position[0]-px)
                        self.angular_velocity = goal_dir - self.heading + np.pi
                    else:
                        self.angular_velocity = 0
            self.last_cluster_prey_num = len(v1)
            self.update_motion(boundary, None)



if __name__ == "__main__":
    agent = Prey(0, np.pi/3, [30, 20])
    fig, ax = plt.subplots()
    ax.add_patch(patches.Circle(
        xy=(agent.position[0], agent.position[1]),
        radius=agent.size,
        fc='lightblue',
        ec='cornflowerblue'
        ))
    ax.grid()
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_aspect(1)
    
    for p in agent.get_odor_sensor_position():
        ax.scatter(p[0], p[1], color='royalblue')
    plt.show()