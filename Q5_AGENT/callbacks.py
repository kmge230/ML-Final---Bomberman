import numpy as np
from collections import defaultdict
from typing import List
import pickle
import os

from .q_learning import Q5TileAgent

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    self.logger.info("Setting up agent.")
    
    # Define hyperparameters
    num_tilings = 5
    num_tiles = 10
    
    # Define features for tile coding
    features = [
        ("x", 0, 16),
        ("y", 0, 16),
        ("bomb_available", 0, 1),
        ("nearest_coin_x", -16, 16),
        ("nearest_coin_y", -16, 16),
        ("nearest_crate_x", -16, 16),
        ("nearest_crate_y", -16, 16),
        ("nearest_opponent_x", -16, 16),
        ("nearest_opponent_y", -16, 16),
        ("in_danger", 0, 1)
    ]
    
    # Initialize Q5-learning agent with tile coding
    self.agent = Q5TileAgent(num_tilings, num_tiles, features, len(ACTIONS))
    
    # Initialize epsilon for epsilon-greedy strategy
    self.epsilon = 0.1 if not self.train else 0.3

    # Load model if it exists and we're not in training mode
    if not self.train and os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.agent.weights = pickle.load(file)
    else:
        self.logger.info("Using a new model.")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    
    if self.train and np.random.rand() < self.epsilon:
        self.logger.debug("Choosing action randomly")
        return np.random.choice(ACTIONS)
    
    self.logger.debug("Querying model for action.")
    return ACTIONS[self.agent.get_action(features)]

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.
    """
    if game_state is None:
        return None

    # Extract relevant information from game state
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    (x, y) = game_state['self'][3]
    bomb_available = game_state['self'][2]
    
    # Calculate features
    nearest_coin = find_nearest_item(field, (x, y), coins)
    nearest_crate = find_nearest_item(field, (x, y), find_crates(field))
    nearest_opponent = find_nearest_item(field, (x, y), [(o[3][0], o[3][1]) for o in game_state['others']])
    in_danger = is_in_danger(field, bombs, explosion_map, (x, y))
    
    features = [
        x, y, int(bomb_available),
        nearest_coin[0] - x if nearest_coin else 0,
        nearest_coin[1] - y if nearest_coin else 0,
        nearest_crate[0] - x if nearest_crate else 0,
        nearest_crate[1] - y if nearest_crate else 0,
        nearest_opponent[0] - x if nearest_opponent else 0,
        nearest_opponent[1] - y if nearest_opponent else 0,
        int(in_danger)
    ]
    
    return np.array(features)

def find_nearest_item(field, pos, items):
    if not items:
        return None
    distances = [manhattan_distance(pos, item) for item in items]
    nearest_index = np.argmin(distances)
    return items[nearest_index]

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_crates(field):
    return [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x, y] == 1]

def is_in_danger(field, bombs, explosion_map, pos):
    for (bx, by), t in bombs:
        if manhattan_distance((bx, by), pos) < 4:
            return True
    return explosion_map[pos[0], pos[1]] > 0

def save_model(self):
    """
    Save the current model to a file.
    """
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.agent.weights, file)

def load_model(self):
    """
    Load the model from a file.
    """
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.agent.weights = pickle.load(file)
        return True
    except:
        self.logger.error("Could not load saved model")
        return False

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']