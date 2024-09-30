import numpy as np
import random

# SARSA agent setup
def setup(self):
    """
    Initialize the agent by setting up necessary parameters for SARSA learning.
    """
    self.alpha = 0.1  # Learning rate
    self.gamma = 0.99  # Discount factor
    self.epsilon = 0.1  # Epsilon-greedy action selection
    self.q_table = {}  # Q-table for SARSA
    self.last_action = None  # Store the last action
    self.last_state = None  # Store the last state
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

# Define the action selection strategy
def act(self, game_state: dict):
    """
    Select an action based on the current game state using an epsilon-greedy policy.
    """
    state = state_to_features(game_state)
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(self.actions))  # Initialize Q-values for unseen states

    if random.uniform(0, 1) < self.epsilon:
        action = random.choice(self.actions)  # Explore: random action
    else:
        action = self.actions[np.argmax(self.q_table[state])]  # Exploit: best known action

    # Store current state and action for future updates
    self.last_state = state
    self.last_action = action

    return action


def state_to_features(game_state: dict):
    """
    Convert the game state to a feature vector.
    This version includes detailed information about the agent's surroundings:
    - Position of coins
    - Distances to the nearest bomb and coin
    - Bomb timers in the vicinity
    - Information about obstacles (walls, crates)
    """
    field = game_state['field']
    x, y = game_state['self'][-1]  # Agent's position
    bombs = game_state['bombs']
    coins = game_state['coins']
    others = [enemy[-1] for enemy in game_state['others']]

    # Feature: Nearest coin
    coin_distances = [np.linalg.norm(np.array((x, y)) - np.array(coin)) for coin in coins]
    nearest_coin_distance = min(coin_distances) if coin_distances else -1  # -1 if no coins visible

    # Feature: Nearest bomb and its timer
    bomb_distances = [np.linalg.norm(np.array((x, y)) - np.array(bomb[0])) for bomb in bombs]
    nearest_bomb_distance = min(bomb_distances) if bomb_distances else -1
    nearest_bomb_timer = bombs[bomb_distances.index(nearest_bomb_distance)][1] if bomb_distances else -1

    # Retrieve the 5-tile surroundings and encode walls, crates, and free space
    def get_5tile_features(x, y, field):
        """
        Retrieve the features for the 5 tiles around the agent.
        """
        features = []
        surroundings = [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x, y)]  # UP, DOWN, LEFT, RIGHT, CURRENT
        for (tx, ty) in surroundings:
            if 0 <= tx < field.shape[0] and 0 <= ty < field.shape[1]:
                features.append(field[tx, ty])  # Append the feature from the field
            else:
                features.append(-1)  # Treat out-of-bound tiles as walls
        return features

    tile_features = get_5tile_features(x, y, field)

    # Combine all features into a single feature vector
    features = tile_features + [nearest_coin_distance, nearest_bomb_distance, nearest_bomb_timer, len(others)]
    return tuple(features)

