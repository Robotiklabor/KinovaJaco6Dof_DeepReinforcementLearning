import pybullet as p
import pybullet_data
import numpy as np
import torch

class Env:

    def __init__(self):
        self.joint_home_Positions=[ 0.000000, 0.000000, 0.087260, 3.3, 1.95, 5.06, 2.52, 0.21, 0.000000, 0.705664, -0.712629, 0.016686, 0.207391, 0.099192, -0.000000 ]
        self.physicsClient = p.connect(p.GUI)    #p.DIRECT for non-graphical version # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.simulation_step_time = 1./240.
        p.setTimeStep(self.simulation_step_time)
        self.planeId = p.loadURDF("plane.urdf")
        self.cup = p.loadURDF("objects/mug.urdf",[-0.5, -0.2, 0.1], p.getQuaternionFromEuler([0,0,0]))
        self.RobotStartPos = [0,0,0]
        self.RobotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.Jaco = p.loadURDF("jaco2.urdf",self.RobotStartPos, self.RobotStartOrientation,flags=p.URDF_USE_SELF_COLLISION)
        
        self.pos_control = False    # p.POSITION_CONTROL == int = 2
        self.vel_control = True     # p.VELOCITY_CONTROL == int = 0
        self.tor_control = False    # p.TORQUE_CONTROL == int = 1
        
        self.max_torques = [20,40,30,20,10,10] # max in urdf: [40,80,40,20,20,20]
        self.step_distance = 0.1
        self.desired_distance_to_goal = 0.1 

    
    def step(self, action):
        # Send action signal to control pipeline for execution
        self.control_pipeline(action)
        # Execute one simulation step
        p.stepSimulation()
        #time.sleep(self.simulation_step_time) # If you want to display the simulation in realtime you can use this line to add a delay
        next_state=self.get_current_state()
        reward=-self.get_distance_tcp_cup()
        done = self.is_close_to_goal()
        return next_state , reward, done

    def control_pipeline(self,action):
        goal = self.get_tcp_position()
        
        if action == 0:
            goal[0] -= self.step_distance
        elif action == 1:
            goal[0] += self.step_distance
        elif action == 2:
            goal[1] -= self.step_distance
        elif action == 3:
            goal[1] += self.step_distance
        elif action == 4:
            goal[2] -= self.step_distance
        elif action == 5:
            goal[2] += self.step_distance
        
        goal = self.inverse_kinematics(goal)
        goal = goal[0:6]
        
        if self.vel_control:
            p.setJointMotorControlArray(self.Jaco, jointIndices=[2,3,4,5,6,7], 
                                controlMode=p.VELOCITY_CONTROL, targetVelocities = goal,
                                forces = self.max_torques)
        elif self.pos_control:
            p.setJointMotorControlArray(self.Jaco, jointIndices=[2,3,4,5,6,7], 
                                controlMode=p.POSITION_CONTROL, targetPositions = goal,
                                forces = self.max_torques)
        elif self.tor_control:
            p.setJointMotorControlArray(self.Jaco, jointIndices=[2,3,4,5,6,7], 
                                controlMode=p.TORQUE_CONTROL, forces = goal)
        else:
            pass

        pass

    def is_in_collision(self):
        collision = False
        contacts = p.getContactPoints(self.Jaco, self.planeId)
        if len(contacts)>1:
            collision = True
        return collision

    def is_in_self_collision(self):
        collisions = p.getContactPoints(self.Jaco, self.Jaco)
        return len(collisions) > 0

    def reset_robot(self):
        for jointIndex in range (p.getNumJoints(self.Jaco)):
            p.resetJointState(self.Jaco,jointIndex,self.joint_home_Positions[jointIndex])
            
    def reset_env(self):
        r = np.random.uniform(0.2, 0.6)
        phi = np.random.uniform(np.pi/2 +0.4, np.pi*3/2 -0.4)
        cup_x = np.cos(phi)*r
        cup_y = np.sin(phi)*r
        p.resetBasePositionAndOrientation(self.cup, [cup_x, cup_y, 0.1], p.getQuaternionFromEuler([0,0,0]))
        for jointIndex in range (p.getNumJoints(self.Jaco)):
            p.resetJointState(self.Jaco,jointIndex,self.joint_home_Positions[jointIndex])
        
    def get_joint_positions(self):
        out = []
        for i in range(2,8):
            out.append(p.getJointState(self.Jaco, jointIndex=i )[0]   )
        return out
    
    def get_joint_velocities(self):
        out = []
        for i in range(2,8):
            out.append(p.getJointState(self.Jaco, jointIndex=i )[1]   )
        return out
    
    def get_joint_torques(self):
        out = []
        for i in range(2,8):
            out.append(p.getJointState(self.Jaco, jointIndex=i )[4]   )
        return out
    
    def get_tcp_position(self):
        out= list(p.getLinkState(self.Jaco, 8)[0])
        return out
    
    def get_tcp_orientation(self):
        out = list(p.getLinkState(self.Jaco, 8)[1])
        return out

    def get_tcp_cart_velocity(self):
        out= list(p.getLinkState(self.Jaco, linkIndex=8, computeLinkVelocity=1)[6])  
        return out

    def get_tcp_angular_velocity(self):
        out= list(p.getLinkState(self.Jaco, linkIndex=8, computeLinkVelocity=1)[7])  
        return out
    
    def get_cup_position(self):
        out = list(p.getBasePositionAndOrientation(self.cup)[0])
        out[2]+=0.1 # Goal-position is 10cm over cup
        return out
    
    def get_cup_orientation(self):
        out = p.getBasePositionAndOrientation(self.cup)[1]
        return out
    
    def get_distance_tcp_cup(self):
        rob = self.get_tcp_position()
        cup =self.get_cup_position()
        dist = np.subtract(cup,rob)
        out = np.linalg.norm(dist)
        return out
    
    def get_current_state(self):
        # To be written dependent of your state-representation
        rob = self.get_tcp_position()
        cup = self.get_cup_position()
        state = np.subtract(cup, rob)
        state=torch.tensor(state,dtype=torch.float32)

        return state
    
    def is_close_to_goal(self):
        close = False
        if self.get_distance_tcp_cup() < self.desired_distance_to_goal:  
            close = True
        return close
    
    def inverse_kinematics(self,position):
        orientation = [1,0,0,0]
        joint_positions = p.calculateInverseKinematics(self.Jaco, 8, position, targetOrientation= orientation)   
        return joint_positions
    
    def enable_position_control(self):
        self.pos_control = True
        self.vel_control = False
        self.tor_control = False
        pass
    def enable_velocity_control(self):
        self.pos_control = False
        self.vel_control = True
        self.tor_control = False
        pass
    def enable_torque_control(self):
        self.pos_control = False
        self.vel_control = False
        self.tor_control = True
        pass
    def get_control_mode(self):
        if self.pos_control:
            return "position_control"
        elif self.vel_control:
            return "velocity_control"
        elif self.tor_control:
            return "torque_control"
        else:
            pass
        
    
    def print_joint_positions(self):
        print("1: %.2f   2: %.2f   3: %.2f   4: %.2f   5: %.2f   6: %.2f" % (p.getJointState(self.Jaco, jointIndex=2)[0] , p.getJointState(self.Jaco, jointIndex=3)[0] ,
        																	 p.getJointState(self.Jaco, jointIndex=4)[0] , p.getJointState(self.Jaco, jointIndex=5)[0] ,
        																	 p.getJointState(self.Jaco, jointIndex=6)[0] , p.getJointState(self.Jaco, jointIndex=7)[0])  )
        