#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import os
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition (Must match Agent)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class DQNTest(Node):
    def __init__(self, stage_num, load_episode):
        super().__init__('dqn_test')

        self.stage = int(stage_num)
        self.load_episode = int(load_episode)
        
        # State size must match what we fixed in Agent (182)
        self.state_size = 182 
        self.action_size = 5

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Test Agent using device: {self.device}")

        # Initialize Model
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        
        # Load Model Weights
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_model'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}.pth'
        )

        if os.path.exists(self.model_path):
            self.get_logger().info(f"Loading model: {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval() # Set to evaluation mode
        else:
            self.get_logger().error(f"Model not found: {self.model_path}")
            self.get_logger().error("Did you train enough episodes? (Default save is every 50)")
            sys.exit(1)

        # ROS Clients
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        # Run for 10 test episodes
        for episode in range(10):
            state = self.reset_environment()
            done = False
            score = 0
            
            while not done:
                # Get Action (Greedy / No Randomness)
                action = self.get_action(state)
                
                # Step
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                
                # Small delay for visualization
                time.sleep(0.01)

            self.get_logger().info(f"Test Episode: {episode+1} | Score: {score}")

    def get_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values).item())

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Environment make client failed to connect...')
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset environment client failed to connect...')
        
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            state = future.result().state
            state = np.array(state).reshape(1, self.state_size)
        else:
            self.get_logger().error('Service call failed')
            state = np.zeros((1, self.state_size))
        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = action
        
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('RL agent interface service not available...')
            
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = np.array(next_state).reshape(1, self.state_size)
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error('Service call failed')
            next_state = np.zeros((1, self.state_size))
            reward = 0
            done = True
            
        return next_state, reward, done

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    load_episode = args[2] if len(args) > 2 else '0'

    rclpy.init(args=args)
    dqn_test = DQNTest(stage_num, load_episode)
    rclpy.spin(dqn_test)
    dqn_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()