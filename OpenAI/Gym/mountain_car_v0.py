import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 1000

# 20 buckets for each observation space action
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# size of each cell range in observation space (each cell are increments of this size)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration probability
epsilon = 0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2

EPSILON_DECAY = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

# 3D Array each channel is an action with 20x20 combinations of position and velocities
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    # get range to find how many increments to get to combination in q_table
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    current_discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[current_discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[current_discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[current_discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[current_discrete_state + (action,)] = 0  # reward for completing

        current_discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= EPSILON_DECAY

env.close()
