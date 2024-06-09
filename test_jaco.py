
import pybullet as p
import numpy as np
from jaco_environment_rl import Env
import torch
import json
from DQN_learner import DQN_learner
import time


def main():

    no_actions = 6
    learn_rate = 0.00000001
    batchsize = 640
    gamma = 0.99
    max_mem_size = 100
    episodes = 100
    episode_length = 1500
    device='cuda'
    score = 0

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    environment = Env()
    environment.reset_robot()
    environment.enable_position_control()
    agent = DQN_learner(no_actions,learn_rate,batchsize,gamma,max_mem_size,device)
    agent.Q = torch.load('my_latest_model.pth')
    

    with open("test_data.json", 'r') as f:
        data = json.load(f)

    for i in range (episodes):
        environment.reset_env()
        episode_steps = 0
        eps=0.01
        p.resetBasePositionAndOrientation(environment.cup, data[i], p.getQuaternionFromEuler([0,0,0])) # Uses position data from test-data-set

        # Episode loop:
        while True:

            state = environment.get_current_state()
            action = agent.choose_action(eps,state)
            next_state, reward, done = environment.step(action)
            episode_steps += 1
            
            if environment.is_in_collision():
                print("Collision with floor")
                break
            if environment.is_in_self_collision():
                print("collision with self")
                break
            if done:
                print("Goal reached")
                score += 1
                break
            elif episode_steps%episode_length == 0:
                print("Episode ended without reaching the goal")
                break

    
    p.disconnect()
    print("Score was: ", score , " of ", episodes)

if __name__== "__main__":
    main()

