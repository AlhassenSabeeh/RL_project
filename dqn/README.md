🤖 TurtleBot3 DQN: Software Engineering & Analysis
This repository contains an optimized Reinforcement Learning implementation for TurtleBot3 using PyTorch, featuring persistent data logging and performance analysis tools.

🛠 Prerequisites
Ensure the following dependencies are installed:

ROS 2 (Humble or Foxy)

Gazebo Simulation Environment

Python Libraries:

Bash
pip install torch torchvision pandas matplotlib numpy pyqtgraph
🚀 Execution Steps
1. Launch the Simulation
Open a terminal to start the Gazebo world.

Bash
export TURTLEBOT3_MODEL=burger
# Use stage 1, 2, 3, or 4
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
2. Start the Environment Node
Open a second terminal. This node calculates the robot's state and rewards.

Bash
python3 dqn_environment.py
3. Run the Training Agent
Open a third terminal. This script handles the Deep Q-Network training and saves data.

Bash
# Usage: python3 dqn_agent.py [stage] [max_episodes]
python3 dqn_agent.py 1 1000
4. Monitor Live Progress
To see real-time reward and Q-value graphs:

Bash
python3 result_graph.py
📊 Post-Training Analysis
Once training is complete (or has reached 200+ episodes), generate the formal report graphs required for the comparison matrix.

Generate Performance Report
Bash
python3 training_analysis.py
This will generate:

performance_report_DQN_Initial.png: A high-resolution plot showing Success Rate, Total Reward, and Training Loss.

saved_model/: Check this folder for the raw .csv data and trained model weights.

📋 Expected Deliverables for Review
Please provide the following feedback after running the simulation:

The generated CSV Log (saved_model/training_log_stage1.csv).

The Performance Graph (performance_report_DQN_Initial.png).

Observation on the Collision Rate during the first 50 episodes.

🔧 Troubleshooting
Resuming Training: Set self.load_model = True in dqn_agent.py to pick up from a saved checkpoint.

Log Errors: If the CSV is not generating, verify you have created the saved_model directory or that the script has write permissions.