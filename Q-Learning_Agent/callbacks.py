import os
import pickle
import random

import numpy as np

#from astar import astar
""" **NOTE that these might be kept in the master, which might not be transferred when we turn in our agent!! """

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    if not hasattr(self, 'q_values'):
        self.q_values = {}

    if self.train:
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        try:
            with open("my-saved-model.pt", "rb") as file:
                self.q_values = pickle.load(file)
        except FileNotFoundError:
            self.logger.info("No saved model found. Starting from scratch.")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    self.logger.debug(f"Choosing action for state: {features}")

    if self.train and np.random.rand() < 0.1:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])

    if features not in self.q_values:
        self.q_values[features] = np.zeros(len(self.actions))

    self.logger.debug("Choosing action based on Q-values.")
    return self.actions[np.argmax(self.q_values[features])]

def state_to_features(game_state: dict) -> tuple:
    """
    Converts the game state to a feature tuple.
    """
    if game_state is None:
        return None

    # Extract relevant information
    _, _, _, (x, y) = game_state['self']
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']

    # Compute features
    nearest_coin = get_nearest_coin_direction(x, y, coins)
    danger = get_danger_level(x, y, bombs, field)
    wall_directions = get_surrounding_tiles(x, y, field)

    return (nearest_coin, danger, wall_directions)

def get_nearest_coin_direction(x, y, coins):
    if not coins:
        return (0, 0)
    nearest_coin = min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    return (np.sign(nearest_coin[0] - x), np.sign(nearest_coin[1] - y))

def get_danger_level(x, y, bombs, field):
    if not bombs:
        return 0
    for (bx, by), t in bombs:
        if abs(bx - x) + abs(by - y) <= 3:
            return 2  # High danger
    return 1  # Low danger

def get_surrounding_tiles(x, y, field):
    return tuple(int(field[x+dx, y+dy]) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)])


"""ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """"""
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """"""
    if self.train:
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.model = None
    elif os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("No saved model found. Setting up model from scratch.")
        print("No saved model found. Setting up model from scratch.")
        self.model = None
    
    
    """"""
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    """"""

def act(self, game_state: dict) -> str:
    """"""
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """"""
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        #print("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])#.1])
    
    features = state_to_features(game_state)
    if self.model is None:
        return np.random.choice(ACTIONS[:4])  # Random movement if no model
    
    q_values = self.model.predict(features.reshape(1, -1))[0]
    return ACTIONS[np.argmax(q_values)]
    

    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """"""
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """"""
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    """""" Begin code for collecting coins """"""
    
    coins = game_state.get('coins')
    arena = game_state.get('field')
    
    player_pos = game_state.get('self')
    
    # Initialize feature vector
    features = []
    
    
    
    # Feature 1-4: Can the agent move in each direction?
    for direction in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # up, right, down, left
        next_pos = (player_pos[3][0] + direction[0], player_pos[3][1] + direction[1])
        features.append(int(arena[next_pos] == 0))
    
    shortpath = np.zeros(40)
    
    
    
    
    # **Submit entire path or just next move?
    
    # Feature 5-6: Direction to the nearest coin
    if coins:
        for coin in coins:
            #print("\ndebug", arena, pos[3], coin)
            path = astar(arena, player_pos[3], coin)
            
            if len(path) < len(shortpath):
                shortpath = path
        if len(shortpath) > 1:
            features.extend(shortpath[1])
            #print(shortpath)
        else:
            features.extend([0, 0])
        shortpath = np.zeros(40)
        
    else:
        features.extend([0, 0])
        
   
    
    return np.array(features)
    
    
    """""" End code for collecting coins """"""
    
    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
"""