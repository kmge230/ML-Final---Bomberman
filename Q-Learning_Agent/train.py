from collections import namedtuple, deque

import pickle
from typing import List

import events as e
#from .callbacks import ACTIONS#state_to_features, ACTIONS

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor

import numpy as np
import random



""" **NOTE that these might be kept in the master, which might not be transferred when we turn in our agent!! """

from collections import defaultdict


from typing import List
from .callbacks import state_to_features

from collections import namedtuple


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.last_features = None
    self.last_action = None

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Convert states to feature tuples
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # Compute the reward
    reward = reward_from_events(self, events)

    # Perform Q-value update
    if old_features is not None:
        if old_features not in self.q_values:
            self.q_values[old_features] = np.zeros(len(self.actions))
        
        old_q = self.q_values[old_features][self.actions.index(self_action)]
        
        if new_features not in self.q_values:
            self.q_values[new_features] = np.zeros(len(self.actions))
        
        next_max_q = np.max(self.q_values[new_features])
        
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
        self.q_values[old_features][self.actions.index(self_action)] = new_q

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Convert state to feature tuple
    last_features = state_to_features(last_game_state)

    # Compute the reward for the final step
    reward = reward_from_events(self, events)

    # Perform a final update of Q-values
    if last_features is not None:
        if last_features not in self.q_values:
            self.q_values[last_features] = np.zeros(len(self.actions))
        
        old_q = self.q_values[last_features][self.actions.index(last_action)]
        new_q = old_q + LEARNING_RATE * (reward - old_q)  # Terminal state, so no future reward
        self.q_values[last_features][self.actions.index(last_action)] = new_q

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_values, file)
    
    #plot_training_data("")

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets to encourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100,
        e.SURVIVED_ROUND: 20,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -3,
        e.INVALID_ACTION: -5,
        e.BOMB_DROPPED: 3,
        e.CRATE_DESTROYED: 5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
            
def plot_training_data(file_path):
    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract round statistics
    rounds = list(data['by_round'].keys())
    steps = [data['by_round'][r]['steps'] for r in rounds]
    coins = [data['by_round'][r]['coins'] for r in rounds]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot coins collected
    ax1.plot(range(1, len(rounds)+1), coins, marker='o')
    ax1.set_title('Coins Collected per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Coins Collected')
    ax1.grid(True)

    # Plot round duration (steps)
    ax2.plot(range(1, len(rounds)+1), steps, marker='o', color='red')
    ax2.set_title('Round Duration')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Number of Steps')
    ax2.grid(True)

    # Add moving averages
    window_size = 20
    if len(rounds) >= window_size:
        coins_ma = np.convolve(coins, np.ones(window_size), 'valid') / window_size
        steps_ma = np.convolve(steps, np.ones(window_size), 'valid') / window_size
        
        ax1.plot(range(window_size, len(rounds)+1), coins_ma, color='orange', linewidth=2, label=f'{window_size}-round Moving Average')
        ax2.plot(range(window_size, len(rounds)+1), steps_ma, color='green', linewidth=2, label=f'{window_size}-round Moving Average')
        
        ax1.legend()
        ax2.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

    print(f"Training progress graph saved as 'training_progress.png'")

    # Calculate and print some statistics
    print(f"\nTraining Statistics:")
    print(f"Total rounds: {len(rounds)}")
    print(f"Average coins per round: {sum(coins) / len(coins):.2f}")
    print(f"Average steps per round: {sum(steps) / len(steps):.2f}")
    print(f"Max coins in a round: {max(coins)}")
    print(f"Min steps in a round: {min(steps)}")
    

"""from helper import plot_learning_curve

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Metrics tracking
METRICS_WINDOW = 10 # no. of episodes to average metrics over

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"



def setup_training(self):
    """"""
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """"""
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    base_model = SGDRegressor(learning_rate='constant', eta0=LEARNING_RATE, random_state=42)
    self.model = MultiOutputRegressor(base_model)
    
    # Initialize model with dummy data
    dummy_X = np.zeros((1, 6))  # 6 features
    dummy_y = np.zeros((1, len(ACTIONS)))  # One output per action
    self.model.fit(dummy_X, dummy_y)
    
    
    self.train_model = train_model.__get__(self)
    
    self.current_score = 0
    self.current_coins = 0
    self.current_step = 0
    self.episode_scores = []
    self.episode_coins = []
    self.episode_steps = []

    
    self.coins_gotten = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features is not None and new_features is not None:
        self.transitions.append(Transition(old_features, self_action, new_features, reward))
        self.train_model()

    # Update current episode metrics
    self.current_score += reward
    self.current_coins += events.count(e.COIN_COLLECTED)
    self.current_step = new_game_state['step']

    """"""
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """"""
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    print(e.COIN_COLLECTED)
    if e.COIN_COLLECTED:
        self.coins_gotten += 1

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    """"""

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    last_features = state_to_features(last_game_state)

    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))
        self.train_model()

    # Update and log metrics
    self.current_score += reward
    self.current_coins += events.count(e.COIN_COLLECTED)
    self.current_step = last_game_state['step']

    self.episode_scores.append(self.current_score)
    self.episode_coins.append(self.current_coins)
    self.episode_steps.append(self.current_step)

    window = min(METRICS_WINDOW, len(self.episode_scores))
    avg_score = sum(self.episode_scores[-window:]) / window
    avg_coins = sum(self.episode_coins[-window:]) / window
    avg_steps = sum(self.episode_steps[-window:]) / window

    self.logger.info(f"Episode {len(self.episode_scores)} metrics:")
    self.logger.info(f"Score: {self.current_score} (Avg over last {len(self.episode_scores)}: {avg_score:.2f})")
    self.logger.info(f"Coins: {self.current_coins} (Avg over last {len(self.episode_coins)}: {avg_coins:.2f})")
    self.logger.info(f"Steps: {self.current_step} (Avg over last {len(self.episode_steps)}: {avg_steps:.2f})")

    # Reset current episode metrics
    self.current_score = 0
    self.current_coins = 0
    self.current_step = 0

    # Save the model
    with open("my-saved-model.pt", "wb") as file:
        #print("Model saved")
        pickle.dump(self.model, file)
        
    
    if len(self.episode_scores) % 200 == 0:
        print("Plot attempt")
        plot_learning_curve(self.episode_scores, self.episode_coins, self.episode_steps)

    """"""
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """"""
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    
    

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
        
    #plot(self.coins_gotten)
    print(self.coins_gotten)
    self.coins_gotten = 0
    """"""

def reward_from_events(self, events: List[str]) -> int:
    """"""
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """"""
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,  # idea: the custom event is bad
        e.GOT_KILLED: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def train_model(self):
    if len(self.transitions) < BATCH_SIZE:
        return

    batch = random.sample(self.transitions, BATCH_SIZE)

    states = np.array([t.state for t in batch])
    next_states = np.array([t.next_state if t.next_state is not None else np.zeros_like(t.state) for t in batch])
    rewards = np.array([t.reward for t in batch])

    current_q_values = self.model.predict(states)
    next_q_values = self.model.predict(next_states)

    for i, (_, action, _, _) in enumerate(batch):
        action_idx = ACTIONS.index(action)
        current_q_values[i, action_idx] = rewards[i] + 0.95 * np.max(next_q_values[i])

    self.model.fit(states, current_q_values)

"""