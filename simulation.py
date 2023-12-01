from threading import Thread
from itertools import combinations
import scipy.io as sio

from matplotlib.pyplot import show

import numpy as np
import cv2

from agent import Prey, Predator, scene3d
from pheromone import Pheromone
from utils import world2image_coordinates_transfer, distance

RESULT_FOLDER = 'results/'

EXP_NAME = 'test'

class Environment:
    def __init__(self, width, height, boundary, dt, obstacles=None):
        self.pheromone = Pheromone(width, height, dt)
        self.pheromone_mask = np.ones(self.pheromone.field.shape)

        self.boundary = boundary
        self.food_catchment = 20
        self.food_position = [self.boundary[0]+self.food_catchment,
                              self.boundary[3]/2]
        
        self.nest_catchment = 20
        self.nest_position = [self.boundary[1]-self.nest_catchment,
                              self.boundary[3]/2]

        if obstacles is not None:
            self.obstacles = obstacles
            # pheromone mask
            for obs in self.obstacles:
                if obs['shape'] == 'rectangle':
                    p1_ = [obs['center'][0] - 0.5*obs['width'], obs['center'][1] + 0.5*obs['height']]
                    p2_ = [obs['center'][0] + 0.5*obs['width'], obs['center'][1] - 0.5*obs['height']]
                    p1 = world2image_coordinates_transfer(p1_,
                                                        self.boundary)
                    p2 = world2image_coordinates_transfer(p2_,
                                                        self.boundary)
                    self.pheromone_mask[p1[0]:p2[0], p1[1]:p2[1]] = 0
        else:
            self.obstacles = None 
            
class Simulator(Thread):
    def __init__(self, time_out, 
                 location, width, height, 
                 obstacles=None, prey_nums=30, 
                 phero_width=20, predator_speed=4,
                 ):
        
        Thread.__init__(self)
        
        self.time = 0
        self.dt = 30 # ms
        self.time_out = time_out
        boundary = [location[0], location[0] + width,
                    location[1], location[1] + height]
        self.environment = Environment(width, height,
                                       boundary, self.dt,
                                       obstacles)

        self.preys = []
        for i in range(prey_nums):
            h = np.random.vonmises(0, 100, 1)[0]
            f = False
            while not f:
                f = True
                pos = [np.random.uniform(30, width-30),
                   np.random.uniform(30, height-30)]
                for b in self.preys:
                    if distance(pos, b.position) <= b.size*2:
                        f = False
                        break
            self.preys.append(Prey(i, h, pos))
        
        self.alive_preys = self.preys.copy()
        self.dead_preys = []
        
        self.predators = []
        # predator releas pheromones
        self.pheromone_inject_k = []
        for i in range(1):
            h = np.random.vonmises(0, 100, 1)[0]
            pos = [width/2, height/2] 
            self.pheromone_inject_k.append([100, 0].copy())
            self.predators.append(Predator(i, h, pos, predator_speed))
        self.pheromone_inject_k = np.array(self.pheromone_inject_k, np.float32)
        self.pheromone_width = phero_width
        
        # visualization
        self.visualize_img = np.zeros((height, width, 3), np.uint8)
        self.visualize_agent_color = {'prey': (135, 206, 235),
                                      'predator':(200, 0, 20)}
        self.visualize_obstacle_color = (125,125,125)
        
        self.cluster = {}
        
        # transfer data
        self.debug_data = {}
        
        # save data
        self.data_to_save = []
        
    
    def get_collide_agents(self, agent:Prey):
        a_ = []
        for a in self.alive_preys:
            if not a.collision:
                a_.append(a)

        co_ = [] # for collision
        co_theta = []
        for a in a_:
            if a is not agent:
                if distance(a.position, agent.position) <= (a.size + agent.size)*1.1:
                    co_.append(a)
                    co_theta.append(np.arctan2((a.position[1] - a.position[1]),
                                               (a.position[0] - a.position[0])))
        return co_, co_theta
    
    def get_near_agents(self, agent:Prey):
        a_ = []
        for a in self.alive_preys:
            if a.cluster_id is None:
                a_.append(a)
                
        cl_ = [] # for cluster
        for a in a_:
            if a is not agent:
                if distance(a.position, agent.position) <= (a.size + agent.size)*2:
                    cl_.append(a)
        return cl_
    
    def arange_cluster(self):
        c_avg = {}
        for ind, ags in self.cluster.items():
            if len(ags) >=2:
                avg_x = np.array([a.position[0] for a in ags]).mean()
                avg_y = np.array([a.position[1] for a in ags]).mean()
                c_avg.update({ind:[avg_x, avg_y]})

        for c_c in combinations(c_avg.keys(), 2):
            if distance(c_avg[c_c[0]], c_avg[c_c[1]]) <= 40:
                # merge cluster
                self.cluster[c_c[0]] += self.cluster[c_c[1]]
                for m_a in self.cluster[c_c[1]]:
                    m_a.cluster_id = c_c[0]
                self.cluster.pop(c_c[1])
    
    def arange_cluster_rectangle(self):
        rect = {}
        margin = 10
        for ind, ags in self.cluster.items():
            if len(ags) >=2:
                x = np.array([a.position[0] for a in ags])
                y = np.array([a.position[1] for a in ags])
                rect.update({ind:[x.min()-margin, x.max()+margin,
                                  y.min()-margin, y.max()+margin]})
        
        for c_c in combinations(rect.keys(), 2):
            r1l = rect[c_c[1]][0]
            r1r = rect[c_c[1]][1]
            r2l = rect[c_c[0]][0]
            r2r = rect[c_c[0]][1]
            r1b = rect[c_c[1]][2]
            r1t= rect[c_c[1]][3]
            r2b = rect[c_c[0]][2]
            r2t = rect[c_c[0]][3]

            if not ((r1l > r2r) or (r1t < r2b) or (r2l > r1r) or (r2t < r1b)):
                for m_a in self.cluster[c_c[1]]:
                    m_a.cluster_id = c_c[0]
            
        self.cluster = {}
        for b in self.alive_preys:
            if b.cluster_id in self.cluster.keys():
                self.cluster[b.cluster_id].append(b)
            elif b.cluster_id is not None:
                self.cluster.update({b.cluster_id:[b]})
                    
    def clear_cluster_collision_info(self):
        self.cluster = {}
        for b in self.alive_preys:
            b.cluster_id = None
            b.collision = False
    
    def visualization2d(self, show_agent=()):
        # pheromone
        self.visualize_img = np.clip(self.environment.pheromone.field,
                                     0, 255).astype(np.uint8)
        # preys
        for a in self.preys:
            p = world2image_coordinates_transfer(a.position,
                                    self.environment.boundary)
            if a.state != 'death':
                self.visualize_img = cv2.circle(self.visualize_img, p[::-1],
                                                a.size,
                                                self.visualize_agent_color['prey'],
                                                thickness=2)
                a_end = [int(a.size*np.cos(a.heading+np.pi/2) + p[0]),
                        int(a.size*np.sin(a.heading+np.pi/2) + p[1])]
                self.visualize_img = cv2.arrowedLine(self.visualize_img,
                                                    p[::-1],
                                                    a_end[::-1],
                                                    (229, 240, 16),
                                                    thickness=2)
                # state
                if a.id in show_agent:
                    state_str = "{}: {}".format(a.id, a.state[0])
                # state_str = "{}: {}/{:.2f}".format(a.id, a.state[0], a.energy)
                    # self.visualize_img = cv2.putText(self.visualize_img, state_str,
                    #                                 (p[1], p[0]-a.size*2), cv2.FONT_HERSHEY_SIMPLEX,
                    #                                 0.5, (255,255,255))
                    cv2.imshow('{}_view'.format(a.id), cv2.cvtColor(a.view, cv2.COLOR_RGB2BGR))
                # state_str = "{}: {},[{}]".format(a.id, a.state[0], a.cluster_id)
                self.visualize_img = cv2.putText(self.visualize_img, str(a.cluster_id),
                                (p[1], p[0]-a.size*2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255))
            else:
                self.visualize_img = cv2.drawMarker(self.visualize_img, p[::-1],
                                                    (100,100,100),
                                                    markerType=cv2.MARKER_CROSS,
                                                    markerSize=a.size)
        # predators
        for a in self.predators:
            p = world2image_coordinates_transfer(a.position,
                                                 self.environment.boundary)
            self.visualize_img = cv2.circle(self.visualize_img, p[::-1],
                                            a.size,
                                            (0,0,255),
                                            thickness=2)
            a_end = [int(a.size*np.cos(a.heading+np.pi/2) + p[0]),
                     int(a.size*np.sin(a.heading+np.pi/2) + p[1])]
            self.visualize_img = cv2.arrowedLine(self.visualize_img,
                                                 p[::-1],
                                                 a_end[::-1],
                                                 (16, 240, 229),
                                                 thickness=2)
    
    def save_experiment_data(self, filename):
        try:
            file = sio.loadmat(filename)
            save_dict = {'data': np.vstack([file['data'], self.data_to_save])}
        except:
            # create a new mat file
            save_dict = {'data':self.data_to_save}
        sio.savemat(filename, save_dict)
        self.data_to_save = []
        
    def run(self, save_data=False, filename='', visualization=False):
        self.time = 0
        end_condition = True if self.time_out is None else (self.time <= self.time_out*1000)
        # alive_agents = filter(lambda p:p.state != 'death', self.preys)
        # visualization parameter
        show_agents = (0, 1)
        while end_condition:
            self.step(save_data, filename)
                    # cv2-based 2D viualization
            if visualization:
                if (self.time % 100 <= self.dt):
                    self.visualization2d(show_agent=show_agents)
                    cv2.imshow('Simulation', cv2.cvtColor(self.visualize_img, cv2.COLOR_RGB2BGR))
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            end_condition = True if self.time_out is None else (self.time <= self.time_out*1000)
        
    def step(self, save_data=False, filename=''):
        # update prey's state
        prey_pos = []
        energy = []
        self.clear_cluster_collision_info()
        for pe in self.alive_preys:
            # cluster
            if pe.cluster_id is None:
                # belong to a new cluster
                a = self.get_near_agents(pe)
                pe.cluster_id = len(self.cluster.keys()) + 1
                self.cluster.update({pe.cluster_id:a + [pe]})
                for a_ in a:
                    a_.cluster_id = pe.cluster_id
            self.arange_cluster_rectangle()
            pe.update(self.dt, self.environment.pheromone.field, self.environment.boundary)
            p = world2image_coordinates_transfer(pe.position,
                                                 self.environment.boundary)
            prey_pos.append(p)
            # update energy
            energy.append(pe.energy)
            if pe.state == 'death':
                if self.predators[-1].state == 'hunting':
                    self.predators[-1].energy += 0.4
                self.dead_preys.append(pe)
                self.alive_preys.remove(pe)
                # spawn a new prey
                h = np.random.vonmises(0, 100, 1)[0]
                pos = [np.random.uniform(30, self.environment.boundary[1] - self.environment.boundary[0] -30),
                        np.random.uniform(30, self.environment.boundary[3] - self.environment.boundary[2] -30)]
                t_prey = Prey(len(self.preys) + 1, h, pos)
                # 1.evolving parameters setted as the average value of the group
                # t_prey.f_gather = np.mean([a.f_gather for a in self.alive_preys])
                # t_prey.f_avoid = np.mean([a.f_avoid for a in self.alive_preys])
                if np.random.uniform(0, 1) > 0.1:
                    # 2.evolving parameters setted with possiblities (agent with max energy has highest possibility)
                    # get alive preys' energy
                    temp_a_e = [(a, a.energy) for a in self.alive_preys]
                    temp_a_e = sorted(temp_a_e, key=lambda x:x[1])
                    sum_ = sum([a_[1] for a_ in temp_a_e])
                    partial_p = [a_[1]/sum_ for a_ in temp_a_e]
                    p_ = np.random.rand(1)[0]
                    ind = np.where(partial_p < p_)[0][-1] if len(np.where(partial_p < p_)[0]) > 0 else -1
                    # print(ind, len(temp_a_e), temp_a_e)
                    t_prey.f_gather = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_gather
                    t_prey.f_avoid = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_avoid
                else:
                    # 3. no evolving, random
                    t_prey.f_gather = np.random.uniform(0.02, 0.32)
                    t_prey.f_avoid = np.random.uniform(0.01, 1.01)
                t_prey.energy = np.mean(np.array([a.energy for a in self.alive_preys]))
                # add to the lists
                self.preys.append(t_prey)
                self.alive_preys.append(t_prey)

        # render pheromone
        predator_pos = []
        for pd in self.predators:
            pd.update(self.environment.boundary,
                        self.cluster, len(self.alive_preys))
            p = world2image_coordinates_transfer(pd.position,
                                                self.environment.boundary)
            predator_pos.append(p)
        self.environment.pheromone.update(predator_pos,
                                          self.pheromone_width,
                                          self.pheromone_inject_k)

        # debug data
        self.debug_data['energy'] = energy
        self.debug_data['f_avoid'] = [a.f_avoid for a in self.alive_preys]
        self.debug_data['f_gather'] = [a.f_gather for a in self.alive_preys]
        self.debug_data['pd_energy'] = self.predators[-1].energy
        
        if save_data:
            # save data
            d = np.zeros([len(self.alive_preys), 7])
            for i, a in enumerate(self.alive_preys):
                d[i] = np.array(([a.id, a.energy, a.f_avoid, a.f_gather,
                                a.position[0], a.position[1], a.heading]))
    
            self.data_to_save.append(list(d))

            if self.time % 100 <= self.dt:
                self.save_experiment_data(filename=filename)
        
        self.time += self.dt


if __name__ == "__main__":
    ARENA_WIDTH = 300
    ARENA_HEIGHT = 300
    ARENA_LOCATION = [0, 0]
    TIMEOUT = None
    
    sim = Simulator(TIMEOUT, ARENA_LOCATION,
                    ARENA_WIDTH, ARENA_HEIGHT, prey_nums=20)
    
    sim.run(visualization=True)
