import gymnasium as gym
import numpy as np
import plotly.graph_objects as go

# Instructions:
# Choose the hyperparameters for the Q-learning algorithm
# To visualise the results, uncomment the line with render_mode="human"
# and run the code.

# --- Hyperparameters ---
ALPHA = [1.0, 0.5, 0.1][2]  # Learning rate
GAMMA = 0.75 # [1.0, 0.5, 0][2]  # Discount factor
EPSILON = [1.0, 0.5, 0][0]  # Initial exploration rate
EPSILON_DECAY = [1.0, 0.95, 0.5][1]  # Decay rate for exploration
MAX_STEPS = [13, 100, 1000][2]  # Max steps per episode
EPISODES = 1000  # Number of training episodes (Fixed)

# Environment setup
env = gym.make("CliffWalking-v0")
# env = gym.make("CliffWalking-v0", render_mode="human")

n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table initialisation
q_table = np.zeros((n_states, n_actions))

# For plotting
episode_rewards = []  # Total rewards per episode
moving_avg_rewards = []  # Average across 100 episodes

epsilon = EPSILON

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + GAMMA * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += ALPHA * td_error

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon *= EPSILON_DECAY

    episode_rewards.append(total_reward)
    # Moving average (window=100)
    if episode >= 99:
        moving_avg = np.mean(episode_rewards[-100:])
        moving_avg_rewards.append(moving_avg)
    else:
        moving_avg_rewards.append(np.mean(episode_rewards))

# Plotting training rewards with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(y=episode_rewards, mode="lines", name="Episode Reward"))
fig.add_trace(
    go.Scatter(y=moving_avg_rewards, mode="lines", name="Moving Avg (100 episodes)")
)
fig.update_layout(
    title="Q-Learning on CliffWalking-v0",
    xaxis_title="Episode",
    yaxis_title="Total Reward",
    legend_title="Legend",
)
fig.show()
