import numpy as np
from typing import List, Tuple

class Q5TileAgent:
    def __init__(self, num_tilings: int, num_tiles: int, features: List[Tuple[str, float, float]], num_actions: int):
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.features = features
        self.num_actions = num_actions
        
        # Initialize 5 sets of weights
        self.weights = [np.ones((num_tilings, num_tiles, len(features), num_actions)) * 15 for _ in range(5)]
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        
    def get_tiles(self, state):
        tiles = []
        for tiling in range(self.num_tilings):
            tile = []
            for i, (_, low, high) in enumerate(self.features):
                offset = tiling / self.num_tilings
                scaled_value = (state[i] - low) / (high - low) * self.num_tiles
                tile_index = int((scaled_value + offset) % self.num_tiles)
                tile.append(tile_index)
            tiles.append(tuple(tile))
        return tiles
    
    def get_q_value(self, state, action, q_index):
        tiles = self.get_tiles(state)
        return sum(self.weights[q_index][t, tiles[t], :, action].sum() for t in range(self.num_tilings))
    
    def get_action(self, state):
        q_values = [max(self.get_q_value(state, a, i) for i in range(5)) for a in range(self.num_actions)]
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state):
        if next_state is None:
            for i in range(5):
                target = reward
                current_q = self.get_q_value(state, action, i)
                td_error = target - current_q
                tiles = self.get_tiles(state)
                for t in range(self.num_tilings):
                    self.weights[i][t, tiles[t], :, action] += self.learning_rate * td_error / self.num_tilings
        else:
            next_q_values = [[self.get_q_value(next_state, a, i) for a in range(self.num_actions)] for i in range(5)]
            for i in range(5):
                max_q = max(next_q_values[i])
                j = np.random.choice([j for j in range(5) if j != i])
                target = reward + self.discount_factor * next_q_values[j][np.argmax(next_q_values[i])]
                current_q = self.get_q_value(state, action, i)
                td_error = target - current_q
                tiles = self.get_tiles(state)
                for t in range(self.num_tilings):
                    self.weights[i][t, tiles[t], :, action] += self.learning_rate * td_error / self.num_tilings