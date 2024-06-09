

import pybullet as p
import numpy as np
from jaco_environment_rl import Env
import time
from DQN_learner import DQN_learner
import torch

def main():
    
    no_actions = 6
    # Hyperparameters
    learn_rate = 0.001
    batchsize = 64
    target_net_sync_frequency = 64
    gamma = 0.99
    max_mem_size = 20000
    eps_start = 0.99
    eps_end = 0.01
    episodes = 400
    episode_length = 1500
    device='cuda'
    check_num = 0
    # Instantiation of environment
    environment = Env()
    environment.enable_position_control()
    DQN_agent = DQN_learner( no_actions, learn_rate, batchsize, gamma, max_mem_size, device)
    epsilon=eps_start

    for i in range (episodes):
        # Reset values at begin of every episode
        environment.reset_env()
        episode_steps = 0
        state=environment.get_current_state()
        if epsilon>=eps_end:
            epsilon-=1.5*(eps_start-eps_end)/episodes
        else:
            epsilon=eps_end

        # Episode loop:
        while True:

            action = DQN_agent.choose_action(epsilon, state)
            next_state, reward, done = environment.step(action)
            check_num += 1
            DQN_agent.save(state, action, reward, next_state, done)
            if check_num >= batchsize:
                states, actions, rewards, next_states, dones = DQN_agent.Sample()
                DQN_agent.Q_update(states, actions, rewards, next_states, dones)
                if check_num % target_net_sync_frequency == 0:
                    DQN_agent.target_net_update()

            state=next_state
            episode_steps += 1
            if done:
                break
            elif episode_steps%episode_length == 0: # Episode ended without reaching the goal
                break
            elif environment.is_in_collision():
                break
            elif environment.is_in_self_collision():
                break

    torch.save(DQN_agent.Q,'model.pth')

    
    # Disconnect pybullet simulation
    p.disconnect()

if __name__== "__main__":
    main()





# Some information regarding pybullet
'''
(p.getJointInfo(Jaco):
    
0 jointIndex
1 jointName
2 jointType
3 Index
4 uIndex
5 flags
6 jointDamping
7 jointFriction
8 jointLowerLimit
9 jointUpperLimit
10 jointMaxForce
11 jointMaxVelocity
12 linkName
13 jointAxis
14 parentFramePos
15 parentFrameOrn
16 parentIndex

JACO:
    
(0, b'connect_root_and_world', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'root', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'j2n6s300_joint_base', 4, -1, -1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b'j2n6s300_link_base', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
(2, b'j2n6s300_joint_1', 0, 7, 6, 1, 0.0, 0.0, -6.283185307179586, 6.283185307179586, 40.0, 0.6283185307179586, b'j2n6s300_link_1', (0.0, 0.0, 1.0), (0.0, 0.0, 0.03125), (0.0, 1.0, 0.0, -6.123233995736766e-17), 1)
(3, b'j2n6s300_joint_2', 0, 8, 7, 1, 0.0, 0.0, 0.8203047484373349, 5.462880558742252, 80.0, 0.6283185307179586, b'j2n6s300_link_2', (0.0, 0.0, 1.0), (0.0, 0.0036, -0.058249999999999996), (-4.329780281177466e-17, -0.7071067811865475, 0.7071067811865476, -4.329780281177467e-17), 2)
(4, b'j2n6s300_joint_3', 0, 9, 8, 1, 0.0, 0.0, 0.33161255787892263, 5.951572749300664, 40.0, 0.6283185307179586, b'j2n6s300_link_3', (0.0, 0.0, 1.0), (0.0, -0.2035, 0.01), (0.0, 1.0, 0.0, -6.123233995736766e-17), 3)
(5, b'j2n6s300_joint_4', 0, 10, 9, 1, 0.0, 0.0, -6.283185307179586, 6.283185307179586, 20.0, 0.8377580409572781, b'j2n6s300_link_4', (0.0, 0.0, 1.0), (0.0, 0.12630000000000002, -0.0028000000000000004), (-4.329780281177466e-17, -0.7071067811865475, 0.7071067811865476, -4.329780281177467e-17), 4)
(6, b'j2n6s300_joint_5', 0, 11, 10, 1, 0.0, 0.0, -6.283185307179586, 6.283185307179586, 20.0, 0.8377580409572781, b'j2n6s300_link_5', (0.0, 0.0, 1.0), (0.0, -3.0000000000002247e-05, 5.9999999999990616e-05), (3.061616997868383e-17, 0.5, 0.8660254037844386, -5.302876193624536e-17), 5)
(7, b'j2n6s300_joint_6', 0, 12, 11, 1, 0.0, 0.0, -6.283185307179586, 6.283185307179586, 20.0, 0.8377580409572781, b'j2n6s300_link_6', (0.0, 0.0, 1.0), (0.0, -3.0000000000002247e-05, 5.9999999999990616e-05), (3.061616997868383e-17, 0.5, 0.8660254037844386, -5.302876193624536e-17), 6)
(8, b'j2n6s300_joint_end_effector', 4, -1, -1, 0, 0.0, 0.0, 0.0, 0.0, 2000.0, 1.0, b'j2n6s300_end_effector', (0.0, 0.0, 0.0), (0.0, 0.0, -0.1), (0.7071067811865476, 0.7071067811865475, 4.329780281177466e-17, -4.329780281177467e-17), 7)
(9, b'j2n6s300_joint_finger_1', 0, 13, 12, 1, 0.0, 0.0, 0.0, 1.51, 2.0, 1.0, b'j2n6s300_link_finger_1', (0.0, 0.0, 1.0), (0.00279, 0.03126, -0.054669999999999996), (0.6629732907414226, 0.2458991985452059, -0.5966991330918943, 0.3794076231254405), 7)
(10, b'j2n6s300_joint_finger_tip_1', 0, 14, 13, 1, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, b'j2n6s300_link_finger_tip_1', (0.0, 0.0, 1.0), (0.022, -0.003, 0.0), (0.0, 0.0, 0.0, 1.0), 9)
(11, b'j2n6s300_joint_finger_2', 0, 15, 14, 1, 0.0, 0.0, 0.0, 1.51, 2.0, 1.0, b'j2n6s300_link_finger_2', (0.0, 0.0, 1.0), (0.02226, -0.02707, -0.05482000000000001), (0.3714595607374613, -0.6016791460724047, 0.2546713192789357, 0.659653332482328), 7)
(12, b'j2n6s300_joint_finger_tip_2', 0, 16, 15, 1, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, b'j2n6s300_link_finger_tip_2', (0.0, 0.0, 1.0), (0.022, -0.003, 0.0), (0.0, 0.0, 0.0, 1.0), 11)
(13, b'j2n6s300_joint_finger_3', 0, 17, 16, 1, 0.0, 0.0, 0.0, 1.51, 2.0, 1.0, b'j2n6s300_link_finger_3', (0.0, 0.0, 1.0), (-0.02226, -0.02707, -0.05482000000000001), (0.25467131926704495, -0.6596533326107493, 0.37145956072557035, 0.6016791459439831), 7)
(14, b'j2n6s300_joint_finger_tip_3', 0, 18, 17, 1, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, b'j2n6s300_link_finger_tip_3', (0.0, 0.0, 1.0), (0.022, -0.003, 0.0), (0.0, 0.0, 0.0, 1.0), 13)

'''