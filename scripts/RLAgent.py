import random
from BlackJackENV import BlackjackEnv
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        action_space,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.9995,
        memory_size=10000,
        batch_size=32,
        target_update=1000,
    ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.05
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update

        # Neural networks
        self.input_size = 6  # State size: (player_sum, dealer_up, has_ace, can_split, can_double, is_blackjack)
        self.output_size = len(action_space)

        self.q_network = DQNNetwork(self.input_size, self.output_size)
        self.target_network = DQNNetwork(self.input_size, self.output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.update_count = 0

    def _state_to_tensor(self, state):
        """Convert state tuple to tensor."""
        if len(state) == 3:  # Old state format
            player_sum, dealer_up, has_ace = state
            return torch.FloatTensor([player_sum, dealer_up, has_ace, 0, 0, 0])
        else:
            return torch.FloatTensor(state)

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Get valid actions based on state for exploration
            valid_actions = self._get_valid_actions(state)
            return random.choice(valid_actions) if valid_actions else 0

        # Get valid actions and choose best Q-value among them
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return 0  # Default to stand if no valid actions

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Only consider valid actions
        valid_q_values = [q_values[action] for action in valid_actions]
        best_valid_idx = valid_q_values.index(max(valid_q_values))
        return valid_actions[best_valid_idx]

    def _get_valid_actions(self, state):
        """Get valid actions based on current state."""
        if (
            len(state) == 3
        ):  # Old state format (player_sum, dealer_up_card, has_usable_ace)
            return [0, 1]  # Only stand and hit for backward compatibility

        (
            player_sum,
            dealer_up_card,
            has_usable_ace,
            can_split,
            can_double,
            is_blackjack,
        ) = state
        valid_actions = [0]  # Stand is always valid

        if player_sum < 21 and not is_blackjack:
            valid_actions.append(1)  # Hit

        if can_double:
            valid_actions.append(2)  # Double down

        if can_split:
            valid_actions.append(3)  # Split

        return valid_actions

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([self._state_to_tensor(s[0]) for s in batch])
        actions = torch.LongTensor([s[1] for s in batch])
        rewards = torch.FloatTensor([s[2] for s in batch])
        next_states = torch.stack([self._state_to_tensor(s[3]) for s in batch])
        dones = torch.BoolTensor([s[4] for s in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filename,
        )

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]

    def train(self, env, episodes, batch_size=32, target_update=100):
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward

            # Decay epsilon after each episode
            self.decay_epsilon()

            if episode % 1000 == 0:
                print(
                    f"Episode {episode}, Epsilon: {self.epsilon:.4f}, Total Reward: {total_reward:.2f}"
                )

    def evaluate(self, env, episodes):
        total_rewards = 0
        total_wins = 0
        # Save current epsilon and set to 0 for pure exploitation during evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(state)
                state, reward, done = env.step(action)
                episode_reward += reward

            total_rewards += episode_reward
            if episode_reward > 0:  # Any positive reward counts as a win
                total_wins += 1

        # Restore original epsilon
        self.epsilon = original_epsilon

        final_win_rate = (total_wins / episodes) * 100
        avg_reward = total_rewards / episodes
        print(f"Final Win Rate: {final_win_rate:.2f}%, Avg Reward: {avg_reward:.3f}")
        return final_win_rate


class QLearningAgent:
    def __init__(
        self,
        action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.999,
    ):
        self.q_table = {}
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Get valid actions based on state for exploration
            valid_actions = self._get_valid_actions(state)
            return random.choice(valid_actions) if valid_actions else 0

        # Get valid actions and choose best Q-value among them
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return 0  # Default to stand if no valid actions

        return max(valid_actions, key=lambda a: self.q_table.get((state, a), 0))

    def _get_valid_actions(self, state):
        """Get valid actions based on current state."""
        if (
            len(state) == 3
        ):  # Old state format (player_sum, dealer_up_card, has_usable_ace)
            return [0, 1]  # Only stand and hit for backward compatibility

        (
            player_sum,
            dealer_up_card,
            has_usable_ace,
            can_split,
            can_double,
            is_blackjack,
        ) = state
        valid_actions = [0]  # Stand is always valid

        if player_sum < 21 and not is_blackjack:
            valid_actions.append(1)  # Hit

        if can_double:
            valid_actions.append(2)  # Double down

        if can_split:
            valid_actions.append(3)  # Split

        return valid_actions

    def update(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)

        # Only consider valid actions for next state
        valid_next_actions = self._get_valid_actions(next_state)
        if valid_next_actions:
            next_max = max(
                [self.q_table.get((next_state, a), 0) for a in valid_next_actions]
            )
        else:
            next_max = 0

        new_value = (1 - self.lr) * old_value + self.lr * (
            reward + self.gamma * next_max
        )
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def evaluate(self, env, episodes):
        total_rewards = 0
        total_wins = 0
        # Save current epsilon and set to 0 for pure exploitation during evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(state)
                state, reward, done = env.step(action)
                episode_reward += reward

            total_rewards += episode_reward
            if episode_reward > 0:  # Any positive reward counts as a win
                total_wins += 1

        # Restore original epsilon
        self.epsilon = original_epsilon

        final_win_rate = (total_wins / episodes) * 100
        avg_reward = total_rewards / episodes
        print(f"Final Win Rate: {final_win_rate:.2f}%, Avg Reward: {avg_reward:.3f}")
        return final_win_rate

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


def train(agent, env, episodes):
    win_rates = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        wins = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        if reward == 1:
            wins += 1
        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            win_rate = (wins / 1000) * 100
            win_rates.append(win_rate)
            wins = 0
            print(
                f"Episode {episode + 1}, Win Rate: {win_rate:.2f}%, Epsilon: {agent.epsilon:.4f}"
            )
    return win_rates