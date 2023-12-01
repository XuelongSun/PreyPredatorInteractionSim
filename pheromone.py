import numpy as np
import cv2

class Pheromone:
    def __init__(self, width, height, dt):
        self.dt = dt
        self.width = width
        self.height = height
        self.field = np.zeros([height, width, 3])
        # pheromone parameters
        self.evaporation = 1e3
        self.diffusion = 0.99
        self.diffusion_kernel = np.array([[(1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8],
                                          [(1 - self.diffusion) / 8,
                                           self.diffusion - 1,
                                           (1 - self.diffusion) / 8],
                                          [(1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8]])
        
    def update(self, inject_pos, inject_size, inject_k):
        for i in range(2):
            # injection
            for pos, k in zip(inject_pos, inject_k):
                
                start_x = pos[0]-inject_size
                start_y = pos[1]-inject_size
                end_x = pos[0]+inject_size
                end_y = pos[1]+inject_size
                
                start_x = max(min(self.width-1, start_x), 0)
                start_y = max(min(self.height-1, start_y), 0)
                end_x = max(min(self.width-1, end_x), 0)
                end_y = max(min(self.height-1, end_y), 0)
                
                self.field[start_x:end_x, start_y:end_y,i]+= k[i]
            
        # evaporation
        self.field += self.field * (-1/self.evaporation) * self.dt
        
        # diffusion
        r, g, b = cv2.split(self.field)
        r = cv2.filter2D(r, -1, self.diffusion_kernel)
        g = cv2.filter2D(g, -1, self.diffusion_kernel)
        b = cv2.filter2D(b, -1, self.diffusion_kernel)
        d = cv2.merge([r, g, b])
        self.field += d*self.dt
        
        self.field = np.clip(self.field, 0.01, 1e4)