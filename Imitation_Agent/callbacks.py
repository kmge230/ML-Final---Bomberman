import os
import pickle
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.ones(len(ACTIONS)) / len(ACTIONS)  # Initialize with equal probabilities
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    
    # Ensure the model contains valid probabilities
    self.model = np.clip(self.model, 1e-7, 1)  # Clip to small positive values
    self.model = self.model / np.sum(self.model)  # Normalize

    if self.train or not os.path.isfile("imitation-model.pt"):
        self.logger.info("Setting up imitation learning model from scratch.")
        self.imitation_model = DecisionTreeClassifier()
        self.imitation_model_fitted = False
    else:
        self.logger.info("Loading imitation learning model from saved state.")
        with open("imitation-model.pt", "rb") as file:
            self.imitation_model = pickle.load(file)
        self.imitation_model_fitted = True

def act(self, game_state: dict) -> str:
    if game_state is None:
        self.logger.warning("Game state is None, choosing action randomly.")
        return np.random.choice(ACTIONS)
    
    features = state_to_features(game_state)
    
    if self.train:
        # Epsilon-greedy strategy for exploration
        epsilon = 0.05
        if random.random() < epsilon:
            self.logger.debug("Choosing action randomly for exploration.")
            return np.random.choice(ACTIONS)
        
        if self.imitation_model_fitted and random.random() < 0.6:  # 70% chance to use imitation learning
            self.logger.debug("Using imitation learning to choose an action.")
            return predict_action_with_imitation_model(self.imitation_model, features)
        else:
            self.logger.debug("Querying model for action.")
            return np.random.choice(ACTIONS, p=self.model)
    else:
        self.logger.debug("Querying model for action.")
        return np.random.choice(ACTIONS, p=self.model)

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    # Extract relevant information from game state
    arena = game_state['field']
    player_position = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    enemies = [agent[3] for agent in game_state['others']]
    explosion_map = game_state['explosion_map']

    # Initialize feature vector
    features = []

    # Feature 1-4: Can move in each direction?
    for direction in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
        new_position = (player_position[0] + direction[0], player_position[1] + direction[1])
        features.append(int(arena[new_position] == 0 and explosion_map[new_position] == 0))

    # Feature 5: Distance to nearest coin
    if coins:
        nearest_coin = min(coins, key=lambda c: manhattan_distance(player_position, c))
        features.append(manhattan_distance(player_position, nearest_coin))
    else:
        features.append(-1)  # No coins available

    # Feature 6: Distance to nearest enemy
    if enemies:
        nearest_enemy = min(enemies, key=lambda e: manhattan_distance(player_position, e))
        features.append(manhattan_distance(player_position, nearest_enemy))
    else:
        features.append(-1)  # No enemies

    # Feature 7: Is bomb available?
    features.append(int(game_state['self'][2]))

    # Feature 8-11: Danger level in each direction
    for direction in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
        danger = 0
        for bomb in bombs:
            bomb_position, timer = bomb
            if manhattan_distance(player_position, bomb_position) <= 3:
                # Check if the direction is towards the bomb
                if (bomb_position[0] - player_position[0]) * direction[0] >= 0 and \
                   (bomb_position[1] - player_position[1]) * direction[1] >= 0:
                    danger = max(danger, 4 - timer)  # Higher danger for bombs about to explode
        features.append(danger)

    # Feature 12: Number of adjacent crates
    adjacent_crates = sum(1 for d in [(0, -1), (1, 0), (0, 1), (-1, 0)] if arena[player_position[0] + d[0], player_position[1] + d[1]] == 1)
    features.append(adjacent_crates)

    # Feature 13: Distance to nearest safe tile after dropping a bomb
    if game_state['self'][2]:  # If bomb is available, consider dropping it
        safe_tiles = get_safe_tiles(arena, bombs + [(player_position, 3)], enemies)
        if safe_tiles:
            nearest_safe = min(safe_tiles, key=lambda pos: manhattan_distance(player_position, pos))
            features.append(manhattan_distance(player_position, nearest_safe))
        else:
            features.append(-1)  # No safe tiles available
    else:
        features.append(-1)  # No bomb available

    return np.array(features)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def predict_action_with_imitation_model(model, features):
    action_index = model.predict([features])[0]
    return ACTIONS[action_index]

def get_safe_tiles(arena, bombs, enemies):
    """Find safe tiles that are not in the bomb blast radius and not occupied by enemies."""
    danger_zones = set()
    for bomb_pos, _ in bombs:
        for i in range(1, 4):
            if arena[bomb_pos[0] + i, bomb_pos[1]] == -1: break
            danger_zones.add((bomb_pos[0] + i, bomb_pos[1]))
        for i in range(1, 4):
            if arena[bomb_pos[0] - i, bomb_pos[1]] == -1: break
            danger_zones.add((bomb_pos[0] - i, bomb_pos[1]))
        for i in range(1, 4):
            if arena[bomb_pos[0], bomb_pos[1] + i] == -1: break
            danger_zones.add((bomb_pos[0], bomb_pos[1] + i))
        for i in range(1, 4):
            if arena[bomb_pos[0], bomb_pos[1] - i] == -1: break
            danger_zones.add((bomb_pos[0], bomb_pos[1] - i))

    safe_tiles = set()
    for x in range(arena.shape[0]):
        for y in range(arena.shape[1]):
            if arena[x, y] == 0 and (x, y) not in danger_zones and (x, y) not in enemies:
                safe_tiles.add((x, y))

    return safe_tiles