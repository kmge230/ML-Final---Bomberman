import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
import numpy as np

from .callbacks import state_to_features, ACTIONS

import matplotlib.pyplot as plt
import json

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# Custom events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_ENEMY = "MOVED_AWAY_FROM_ENEMY"
BOMB_DROPPED_NEAR_CRATE = "BOMB_DROPPED_NEAR_CRATE"
ESCAPED_EXPLOSION = "ESCAPED_EXPLOSION"
MOVED_TO_SAFETY_AFTER_BOMB = "MOVED_TO_SAFETY_AFTER_BOMB"

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.num_rounds = 0
    
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.imitation_data = []
    self.enemy_transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    # Initialize self.model if it doesn't exist
    if not hasattr(self, 'model') or self.model is None:
        self.model = np.ones(len(ACTIONS)) / len(ACTIONS)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
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
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Add custom events
    events.extend(get_custom_events(old_game_state, self_action, new_game_state))

    # Store transition
    if old_game_state and new_game_state:
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)
        if old_features is not None and new_features is not None:
            self.transitions.append(Transition(
                old_features,
                self_action,
                new_features,
                reward_from_events(self, events)
            ))

    # Store imitation learning data
    if old_game_state:
        features = state_to_features(old_game_state)
        if features is not None and self_action in ACTIONS:
            self.imitation_data.append((features, ACTIONS.index(self_action)))
        else:
            self.logger.warning(f"Skipping imitation data storage. Features: {features}, Action: {self_action}")

    self.logger.debug(f'Encountered {len(events)} game event(s)')

def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str, enemy_game_state: dict, enemy_events: List[str]):
    """
    Called once per step to allow training on enemy actions.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param enemy_name: The name of the enemy
    :param old_enemy_game_state: The state before the enemy acted.
    :param enemy_action: The enemy's chosen action.
    :param enemy_game_state: The state after the enemy acted.
    :param enemy_events: The events that occurred due to the enemy's actions.
    """
    self.logger.debug(f'Encountered enemy game event(s) for {enemy_name}: {", ".join(map(repr, enemy_events))} in step {enemy_game_state["step"]}')

    # Store enemy transition
    if old_enemy_game_state and enemy_game_state:
        old_features = state_to_features(old_enemy_game_state)
        new_features = state_to_features(enemy_game_state)
        if old_features is not None and new_features is not None:
            self.enemy_transitions.append(Transition(
                old_features,
                enemy_action,
                new_features,
                reward_from_events(self, enemy_events)
            ))

    # Store enemy action for imitation learning
    if old_enemy_game_state:
        features = state_to_features(old_enemy_game_state)
        if features is not None and enemy_action in ACTIONS:
            self.imitation_data.append((features, ACTIONS.index(enemy_action)))
        else:
            self.logger.warning(f"Skipping imitation data storage for enemy {enemy_name}. Features: {features}, Action: {enemy_action}")

    self.logger.debug(f'Stored enemy transition and imitation data for {enemy_name}')

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Add custom events for the final step
    events.extend(get_custom_events(last_game_state, last_action, None))

    # Store the last transition
    if last_game_state:
        features = state_to_features(last_game_state)
        if features is not None:
            self.transitions.append(Transition(
                features,
                last_action,
                None,
                reward_from_events(self, events)
            ))

    # Ensure self.model is properly initialized
    if not hasattr(self, 'model') or self.model is None:
        self.logger.warning("self.model was not initialized. Initializing now.")
        self.model = np.ones(len(ACTIONS)) / len(ACTIONS)

    # Train your model here
    all_transitions = list(self.transitions) + list(self.enemy_transitions)
    for t in all_transitions:
        if t.state is not None and t.action in ACTIONS:
            action_index = ACTIONS.index(t.action)
            current_q = self.model[action_index]
            if t.next_state is not None:
                best_next_action = ACTIONS[np.argmax(self.model)]
                max_future_q = self.model[ACTIONS.index(best_next_action)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (t.reward + DISCOUNT_FACTOR * max_future_q)
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * t.reward
            self.model[action_index] = max(1e-7, new_q)  # Ensure non-negative and non-zero

    # Normalize the model
    self.model = self.model / np.sum(self.model)

    # Clip values to prevent extreme probabilities
    self.model = np.clip(self.model, 1e-7, 1)
    self.model = self.model / np.sum(self.model)  # Renormalize after clipping

    # Train imitation learning model
    if self.imitation_data:
        X, y = zip(*self.imitation_data)
        self.imitation_model.fit(X, y)
        self.imitation_model_fitted = True

    # Store the models
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    
    if hasattr(self, 'imitation_model'):
        with open("imitation-model.pt", "wb") as file:
            pickle.dump(self.imitation_model, file)

    # Clear the transitions and imitation data for the next round
    self.transitions.clear()
    self.enemy_transitions.clear()
    self.imitation_data.clear()
    
    # Calculate the score for this round
    score = reward_from_events(self, events)

    # Update and save training data
    if not hasattr(self, 'training_data'):
        self.training_data = {'by_round': {}}
    
    round_number = len(self.training_data['by_round']) + 1
    self.training_data['by_round'][str(round_number)] = {
        'steps': int(last_game_state['step']),
        'coins': int(last_game_state['self'][1]),
        'score': int(score)
    }
    
    with open("training_data.json", "w") as f:
        json.dump(self.training_data, f)
        
    self.num_rounds += 1

    if self.num_rounds % 100 == 0:
        plot_training_data("training_data.json")

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 20,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -15,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 1,
        e.SURVIVED_ROUND: 10,
        MOVED_TOWARDS_COIN: 1,
        e.INVALID_ACTION: -1,
        e.WAITED: -3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    # Clip reward to prevent extreme values
    reward_sum = np.clip(reward_sum, -100, 100)
    
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def get_custom_events(old_game_state: dict, action: str, new_game_state: dict) -> List[str]:
    custom_events = []

    if old_game_state and new_game_state:
        old_coins = old_game_state['coins']
        new_coins = new_game_state['coins']
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Check if moved towards a coin
        if old_coins and new_coins:
            old_nearest_coin_distance = min(manhattan_distance(old_position, coin) for coin in old_coins)
            new_nearest_coin_distance = min(manhattan_distance(new_position, coin) for coin in new_coins)
            if new_nearest_coin_distance < old_nearest_coin_distance:
                custom_events.append(MOVED_TOWARDS_COIN)

        # Check if moved away from an enemy
        old_enemies = [agent[3] for agent in old_game_state['others']]
        new_enemies = [agent[3] for agent in new_game_state['others']]
        if old_enemies and new_enemies:
            old_nearest_enemy_distance = min(manhattan_distance(old_position, enemy) for enemy in old_enemies)
            new_nearest_enemy_distance = min(manhattan_distance(new_position, enemy) for enemy in new_enemies)
            if new_nearest_enemy_distance > old_nearest_enemy_distance:
                custom_events.append(MOVED_AWAY_FROM_ENEMY)

        # Check if bomb was dropped near a crate
        if action == 'BOMB':
            arena = new_game_state['field']
            if any(arena[new_position[0] + dx, new_position[1] + dy] == 1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                custom_events.append(BOMB_DROPPED_NEAR_CRATE)

        # Check if agent escaped an explosion
        old_bombs = old_game_state['bombs']
        new_bombs = new_game_state['bombs']
        if escaped_explosion(old_position, new_position, old_bombs, new_bombs):
            custom_events.append(ESCAPED_EXPLOSION)

        # Check if agent moved to safety after dropping a bomb
        if old_game_state['self'][2] and not new_game_state['self'][2]:  # Bomb was just placed
            if len(new_bombs) > len(old_bombs):  # Confirm a new bomb was added
                last_bomb = new_bombs[-1]
                if manhattan_distance(new_position, last_bomb[0]) > manhattan_distance(old_position, last_bomb[0]):
                    # Agent moved away from the bomb
                    if is_behind_corner_or_crate(new_position, last_bomb[0], new_game_state['field']):
                        custom_events.append(MOVED_TO_SAFETY_AFTER_BOMB)

    return custom_events

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def escaped_explosion(old_pos, new_pos, old_bombs, new_bombs):
    # Check if the agent moved away from a bomb that's about to explode
    for old_bomb in old_bombs:
        if old_bomb not in new_bombs:  # The bomb exploded
            bomb_pos, timer = old_bomb
            if manhattan_distance(old_pos, bomb_pos) <= 3 and manhattan_distance(new_pos, bomb_pos) > 3:
                return True
    return False

def is_behind_corner_or_crate(agent_pos, bomb_pos, field):
    # Check if there's a corner or crate between the agent and the bomb
    dx = agent_pos[0] - bomb_pos[0]
    dy = agent_pos[1] - bomb_pos[1]
    
    if dx != 0 and dy != 0:  # Diagonal movement, already safe
        return True
    
    if dx != 0:
        check_x = bomb_pos[0] + np.sign(dx)
        return field[check_x, agent_pos[1]] != 0  # Not free space
    
    if dy != 0:
        check_y = bomb_pos[1] + np.sign(dy)
        return field[agent_pos[0], check_y] != 0  # Not free space
    
    return False  # Agent didn't move

def plot_training_data(file_path):
    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract round statistics
    rounds = list(data['by_round'].keys())
    steps = [data['by_round'][r]['steps'] for r in rounds]
    coins = [data['by_round'][r]['coins'] for r in rounds]
    scores = [data['by_round'][r]['score'] for r in rounds]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Calculate averages after 1000 iterations
    avg_coins = np.mean(coins[1000:]) if len(coins) > 1000 else np.mean(coins)
    avg_steps = np.mean(steps[1000:]) if len(steps) > 1000 else np.mean(steps)
    avg_scores = np.mean(scores[1000:]) if len(scores) > 1000 else np.mean(scores)

    # Plot coins collected
    ax1.plot(range(1, len(rounds)+1), coins, marker='o')
    ax1.set_title('Coins Collected per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Coins Collected')
    ax1.grid(True)
    ax1.text(1.02, 0.5, f'Avg after 1000: {avg_coins:.2f}', transform=ax1.transAxes, verticalalignment='center')

    # Plot round duration (steps)
    ax2.plot(range(1, len(rounds)+1), steps, marker='o', color='red')
    ax2.set_title('Round Duration')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Number of Steps')
    ax2.grid(True)
    ax2.text(1.02, 0.5, f'Avg after 1000: {avg_steps:.2f}', transform=ax2.transAxes, verticalalignment='center')

    # Plot scores
    ax3.plot(range(1, len(rounds)+1), scores, marker='o', color='green')
    ax3.set_title('Reward per Round')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.text(1.02, 0.5, f'Avg after 1000: {avg_scores:.2f}', transform=ax3.transAxes, verticalalignment='center')

    # Add moving averages
    window_size = 20
    if len(rounds) >= window_size:
        coins_ma = np.convolve(coins, np.ones(window_size), 'valid') / window_size
        steps_ma = np.convolve(steps, np.ones(window_size), 'valid') / window_size
        scores_ma = np.convolve(scores, np.ones(window_size), 'valid') / window_size
        
        ax1.plot(range(window_size, len(rounds)+1), coins_ma, color='orange', linewidth=2, label=f'{window_size}-round Moving Average')
        ax2.plot(range(window_size, len(rounds)+1), steps_ma, color='purple', linewidth=2, label=f'{window_size}-round Moving Average')
        ax3.plot(range(window_size, len(rounds)+1), scores_ma, color='blue', linewidth=2, label=f'{window_size}-round Moving Average')
        
        ax1.legend()
        ax2.legend()
        ax3.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

# You can add any additional helper functions or constants here if needed