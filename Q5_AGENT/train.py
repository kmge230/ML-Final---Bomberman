import events as e
from .callbacks import state_to_features, save_model, ACTIONS
from typing import List

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = []
    self.reward_sum = 0
    self.reward_map = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.1,
        e.BOMB_DROPPED: 0.1,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.3,
        e.SURVIVED_ROUND: 1,
    }

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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(str, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        return
    
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    
    self.transitions.append((old_features, ACTIONS.index(self_action), reward, new_features))
    self.reward_sum += reward
    
    # Perform Q5-learning update
    self.agent.update(old_features, ACTIONS.index(self_action), reward, new_features)

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(str, events))} in final step')
    
    # Add the last transition
    old_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.transitions.append((old_features, ACTIONS.index(last_action), reward, None))
    self.reward_sum += reward
    
    # Perform final Q5-learning update
    self.agent.update(old_features, ACTIONS.index(last_action), reward, None)
    
    # Log episode statistics
    self.logger.info(f"Episode reward: {self.reward_sum}")
    
    # Reset episode-specific variables
    self.transitions = []
    self.reward_sum = 0
    
    # Decay epsilon
    self.epsilon = max(0.05, self.epsilon * 0.99)

    # Save the model
    save_model(self)

    self.logger.info("Model saved successfully.")

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in self.reward_map:
            reward_sum += self.reward_map[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum