import numpy as np
import events as e
from .callbacks import state_to_features

def setup_training(self):
    """
    Initialize variables needed for training.
    """
    self.transition_buffer = []  # Buffer to store transitions for SARSA updates
    self.epsilon = 1.0

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    This method is called after each step to track what events occurred and to update the Q-values.
    """
    # Extract the features of the old and new game state
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    # Reward shaping based on events
    reward = reward_from_events(self, events)

    # SARSA Q-value update
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(self.actions))  # Initialize unseen state in Q-table

    # Next action using epsilon-greedy
    if random.uniform(0, 1) < self.epsilon:
        next_action = random.choice(self.actions)  # Explore
    else:
        next_action = self.actions[np.argmax(self.q_table[new_state])]  # Exploit

    # Q-value update for SARSA
    old_q_value = self.q_table[old_state][self.actions.index(self_action)]
    next_q_value = self.q_table[new_state][self.actions.index(next_action)]
    self.q_table[old_state][self.actions.index(self_action)] = old_q_value + self.alpha * (
        reward + self.gamma * next_q_value - old_q_value
    )

    # Store the last state-action pair
    self.last_state = new_state
    self.last_action = next_action

def end_of_round(self, last_game_state, last_action, events):
    """
    This method is called at the end of each round to finalize the learning updates.
    """
    # Handle the end-of-round events
    reward = reward_from_events(self, events)
    last_state = state_to_features(last_game_state)

    # Update the final Q-value for the last action
    old_q_value = self.q_table[last_state][self.actions.index(last_action)]
    self.q_table[last_state][self.actions.index(last_action)] = old_q_value + self.alpha * (
        reward - old_q_value  # No future state, so the next Q-value is 0
    )

    # Decay epsilon after each round to reduce exploration over time
    update_epsilon(self)

def update_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
    """
    Decay epsilon over time to reduce exploration.
    """
    self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def reward_from_events(self, events):
    """
    Calculate the reward from events that occurred during the step.
    """
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 10  # Encourage coin collection
    if e.MOVED_LEFT in events or e.MOVED_RIGHT in events or e.MOVED_UP in events or e.MOVED_DOWN in events:
        reward += 1  # Small reward for movement (to encourage exploration)
    if e.CRATE_DESTROYED in events:
        reward += 5  # Encourage destroying crates
    if e.KILLED_OPPONENT in events:
        reward += 20  # High reward for killing an opponent
    if e.KILLED_SELF in events:
        reward -= 20  # High penalty for killing self
    if e.GOT_KILLED in events:
        reward -= 10  # Penalty for getting killed by opponents
    if e.INVALID_ACTION in events:
        reward -= 1  # Penalize invalid actions
    return reward



import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progress Over Time')
    plt.show()