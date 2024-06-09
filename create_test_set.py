
import numpy as np
import json

data = []

for i in range(100):
    r = np.random.uniform(0.25, 0.6)
    phi = np.random.uniform(np.pi/2 +0.4, np.pi*3/2 -0.4)
    goal_x = np.cos(phi)*r
    goal_y = np.sin(phi)*r
    goal_z = 0.1
    
    data.append([goal_x, goal_y, goal_z])

with open("test_data.json", 'w') as f:
    json.dump(data, f, indent=2) 

