import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_training(csv_file, algo_name="DQN"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Start training first!")
        return

    # Load the data
    df = pd.read_csv(csv_file)

    # 1. Define Success/Collision (Simplified Logic)
    # Usually, a very high reward means 'Goal' and a very low reward means 'Collision'
    success_threshold = 100  # Adjust based on your dqn_environment.py rewards
    df['is_success'] = df['Total_Reward'] >= success_threshold
    
    # Calculate Success Rate over a rolling window of 50 episodes
    df['Success_Rate'] = df['is_success'].rolling(window=50).mean() * 100

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # --- Plot A: Total Reward & Success Rate ---
    ax1.plot(df['Episode'], df['Total_Reward'], alpha=0.3, color='blue', label=f'{algo_name} Raw Reward')
    ax1.plot(df['Episode'], df['Total_Reward'].rolling(window=20).mean(), color='blue', linewidth=2, label=f'{algo_name} Moving Avg')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Performance & Success Rate')
    
    # Add Success Rate on second Y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Episode'], df['Success_Rate'], color='green', linestyle='--', label='Success Rate (%)')
    ax1_twin.set_ylabel('Success Rate (%)')
    ax1_twin.set_ylim(0, 100)
    
    # --- Plot B: Loss & Q-Value (The "Brain" Metrics) ---
    ax2.plot(df['Episode'], df['Loss'], color='red', alpha=0.5, label='Training Loss')
    ax2.set_yscale('log') # Loss is often better viewed on a log scale
    ax2.set_ylabel('Loss (Log Scale)')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Episode'], df['Avg_Max_Q'], color='purple', label='Avg Max Q (Confidence)')
    ax2_twin.set_ylabel('Q-Value Confidence')

    plt.xlabel('Episode')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'performance_report_{algo_name}.png')
    print(f"Report saved as performance_report_{algo_name}.png")
    plt.show()

# Run the analysis
analyze_training('saved_model/training_log_stage1.csv', algo_name="DQN_Initial")