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

import collections
import datetime
import json
import math
import os
import random
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
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------------------------------------------------------------
# PyTorch Model Definition
# ----------------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, action_size)

        # Weight initialization (optional, helps convergence)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

# ----------------------------------------------------------------------------------
# DQN Agent Node
# ----------------------------------------------------------------------------------
class DQNAgent(Node):
    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent')

        # --- Parameters ---
        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 182
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)
        
        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 6000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 64  # Typical batch size for DQN

        # Experience Replay
        self.replay_memory = collections.deque(maxlen=1000000)
        self.min_replay_memory_size = 64

        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Networks ---
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        
        # Optimizer & Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()
        self.update_target_after = 2000  # steps
        self.target_update_after_counter = 0

        # --- Model Loading/Saving ---
        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_model'
        )
        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)

        self.model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}.pth'
        )

        if self.load_model:
            if os.path.isfile(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.update_target_model()
                # Load JSON for epsilon/steps
                json_path = self.model_path.replace('.pth', '.json')
                if os.path.isfile(json_path):
                    with open(json_path) as outfile:
                        param = json.load(outfile)
                        self.epsilon = param.get('epsilon')
                        self.step_counter = param.get('step_counter')
                self.get_logger().info(f"Loaded model: {self.model_path}")
            else:
                self.get_logger().warn(f"Model file not found: {self.model_path}")

        # --- ROS 2 Interfaces ---
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, 'action_graph', 10) # fixed topic name
        self.result_pub = self.create_publisher(Float32MultiArray, 'result_graph', 10) # fixed topic name

        # Start Process
        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0
            
            # Start Episode
            done = False
            while not done:
                local_step += 1

                # 1. Action Selection
                action = int(self.get_action(state))

                # 2. Step Environment (Service Call)
                next_state, reward, done = self.step(action)
                
                score += reward

                # 3. Publish Action/Score for Graphs
                # We do this before training to visualize real-time
                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                # 4. Training Step
                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)
                    
                    # Accumulate Max Q for stats
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).to(self.device)
                        q_values = self.model(state_t)
                        sum_max_q += torch.max(q_values).item()

                state = next_state

                # 5. Episode End Handling
                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0
                    
                    # Publish Result for Graphs
                    res_msg = Float32MultiArray()
                    res_msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(res_msg)

                    self.get_logger().info(
                        f"Episode: {episode} | Score: {score:.2f} | "
                        f"Steps: {local_step} | Epsilon: {self.epsilon:.4f} | "
                        f"Memory: {len(self.replay_memory)}"
                    )

                    # Save Model every 50 episodes
                    if self.train_mode and episode % 10 == 0:
                        self.save_model(episode)
                    break
                
                # Small sleep to prevent busy loop if simulation is slow
                # time.sleep(0.001) 

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
            self.get_logger().error(f'Service call failed: {future.exception()}')
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
            self.get_logger().error(f'Service call failed: {future.exception()}')
            next_state = np.zeros((1, self.state_size))
            reward = 0.0
            done = True

        return next_state, reward, done

    def get_action(self, state):
        # Epsilon Decay
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + \
                (1.0 - self.epsilon_min) * math.exp(-1.0 * self.step_counter / self.epsilon_decay)

        # Epsilon Greedy
        if self.train_mode and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(self.device)
                q_values = self.model(state_t)
                return torch.argmax(q_values).item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.get_logger().info('Target model updated')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Sample Batch
        batch = random.sample(self.replay_memory, self.batch_size)
        
        # Prepare Tensors
        states      = torch.FloatTensor(np.array([t[0] for t in batch])).squeeze(1).to(self.device)
        actions     = torch.LongTensor(np.array([t[1] for t in batch])).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).squeeze(1).to(self.device)
        dones       = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1).to(self.device)

        # 1. Current Q Values
        current_q = self.model(states).gather(1, actions)

        # 2. Target Q Values (Bellman Equation)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.discount_factor * max_next_q * (1 - dones))

        # 3. Compute Loss & Optimize
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 4. Soft/Hard Target Update
        self.target_update_after_counter += 1
        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()
            self.target_update_after_counter = 0

    def save_model(self, episode):
        save_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{episode}.pth'
        )
        torch.save(self.model.state_dict(), save_path)
        
        # Save JSON for parameters
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as outfile:
            json.dump({'epsilon': self.epsilon, 'step_counter': self.step_counter}, outfile)
        
        self.get_logger().info(f'Model saved: {save_path}')

def main(args=None):
    if args is None:
        args = sys.argv
    
    # Defaults if args missing
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'

    rclpy.init(args=args)
    dqn_agent = DQNAgent(stage_num, max_training_episodes)
    
    try:
        rclpy.spin(dqn_agent)
    except KeyboardInterrupt:
        pass
    finally:
        dqn_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()