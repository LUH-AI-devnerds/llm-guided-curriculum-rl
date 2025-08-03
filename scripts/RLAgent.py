import random
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
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


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
        curriculum_stage=None,
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
        self.curriculum_stage = curriculum_stage

        self.input_size = 11
        self.output_size = len(action_space)

        self.q_network = DQNNetwork(self.input_size, self.output_size)
        self.target_network = DQNNetwork(self.input_size, self.output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.update_count = 0

    def set_curriculum_stage(self, curriculum_stage):
        old_stage_id = self.curriculum_stage.stage_id if self.curriculum_stage else 0
        self.curriculum_stage = curriculum_stage

        if curriculum_stage and curriculum_stage.stage_id > old_stage_id:
            # Keep more experiences for better knowledge transfer
            keep_size = max(2000, int(len(self.memory) * 0.4))
            if len(self.memory) > keep_size:
                recent_experiences = list(self.memory)[-keep_size:]
                self.memory.clear()
                self.memory.extend(recent_experiences)
                print(
                    f"  🧹 Cleaned memory: kept {len(self.memory)} recent experiences"
                )

        # Always reset epsilon for current stage (not just when advancing)
        if curriculum_stage and curriculum_stage.stage_id > 1:
            # Progressive epsilon reduction based on stage complexity
            if curriculum_stage.stage_id == 2:
                self.epsilon = max(0.2, self.epsilon)  # Double down stage
            elif curriculum_stage.stage_id == 3:
                self.epsilon = max(0.15, self.epsilon)  # Split stage
            elif curriculum_stage.stage_id == 4:
                self.epsilon = max(0.1, self.epsilon)  # Surrender stage
            elif curriculum_stage.stage_id >= 5:
                self.epsilon = max(0.05, self.epsilon)  # Insurance stage
            print(
                f"  🔄 Reset epsilon to {self.epsilon:.3f} for stage {curriculum_stage.stage_id}"
            )

            if curriculum_stage.stage_id >= 3:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = min(param_group["lr"] * 1.2, 0.005)
                print(
                    f"  📈 Increased learning rate to {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        if curriculum_stage and curriculum_stage.stage_id == 4:
            print("  🔧 Initializing conservative Q-values for Split action")
            with torch.no_grad():
                if hasattr(self.q_network, "fc5"):
                    self.q_network.fc5.bias[3] -= 0.5

    def _state_to_tensor(self, state):
        if len(state) == 3:
            player_sum, dealer_up, has_ace = state
            full_state = [player_sum, dealer_up, int(has_ace), 0, 0, 0, 0, 0, 0, 0, 0]
        elif len(state) == 6:
            full_state = list(state) + [0, 0, 0, 0, 0]
        elif len(state) == 9:
            full_state = list(state) + [0, 0]
        elif len(state) >= 11:
            full_state = []
            for i in range(11):
                val = state[i] if i < len(state) else 0
                if val is None:
                    val = 0
                elif isinstance(val, bool):
                    val = int(val)
                full_state.append(float(val))
        else:
            full_state = list(state) + [0] * (11 - len(state))

        for i in range(len(full_state)):
            if not isinstance(full_state[i], (int, float)) or not np.isfinite(
                full_state[i]
            ):
                full_state[i] = 0.0

        return torch.FloatTensor(full_state)

    def _get_learning_state(self, state):
        if len(state) <= 9:
            return state
        else:
            return state[:9]

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            valid_actions = self._get_valid_actions(state)

            if self.curriculum_stage and self.curriculum_stage.stage_id >= 3:
                if 2 in valid_actions and random.random() < 0.3:
                    return 2
                if (
                    self.curriculum_stage.stage_id == 4
                    and 3 in valid_actions
                    and random.random() < 0.2
                ):
                    return 3
                if (
                    self.curriculum_stage.stage_id >= 4
                    and 4 in valid_actions
                    and random.random() < 0.1
                ):
                    return 4
                if (
                    self.curriculum_stage.stage_id >= 5
                    and 5 in valid_actions
                    and random.random() < 0.1
                ):
                    return 5

            return random.choice(valid_actions) if valid_actions else 0

        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return 0

        learning_state = self._get_learning_state(state)
        state_tensor = self._state_to_tensor(learning_state)
        state_tensor = state_tensor.unsqueeze(0)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.squeeze(0)
        self.q_network.train()

        valid_q_values = [q_values[action] for action in valid_actions]
        best_valid_idx = valid_q_values.index(max(valid_q_values))
        return valid_actions[best_valid_idx]

    def _get_valid_actions(self, state):
        if len(state) == 3:
            base_actions = [0, 1]
        else:
            player_sum = state[0] if len(state) > 0 else 0
            dealer_up_card = state[1] if len(state) > 1 else 0
            has_usable_ace = state[2] if len(state) > 2 else False
            can_split = state[3] if len(state) > 3 else False
            can_double = state[4] if len(state) > 4 else False
            is_blackjack = state[5] if len(state) > 5 else False
            can_surrender = state[6] if len(state) > 6 else False
            can_insure = state[7] if len(state) > 7 else False

            base_actions = [0]

            if player_sum < 21 and not is_blackjack:
                base_actions.append(1)

            if can_double:
                base_actions.append(2)

            if can_split:
                base_actions.append(3)

            if can_surrender:
                base_actions.append(4)

            if can_insure:
                base_actions.append(5)

        if self.curriculum_stage is not None:
            valid_actions = [
                a for a in base_actions if a in self.curriculum_stage.available_actions
            ]
            return valid_actions if valid_actions else [0]
        else:
            return base_actions

    def remember(self, state, action, reward, next_state, done):
        learning_state = self._get_learning_state(state)
        learning_next_state = self._get_learning_state(next_state)
        self.memory.append((learning_state, action, reward, learning_next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        if self.curriculum_stage and self.curriculum_stage.stage_id > 1:
            recent_size = min(len(self.memory) // 2, self.batch_size // 2)
            recent_batch = random.sample(
                list(self.memory)[-len(self.memory) // 2 :], recent_size
            )
            older_batch = random.sample(self.memory, self.batch_size - recent_size)
            batch = recent_batch + older_batch
        else:
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

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.curriculum_stage:
            # Faster decay for advanced stages
            stage_factor = 1.0 + (self.curriculum_stage.stage_id - 1) * 0.2
            adjusted_decay = self.epsilon_decay**stage_factor
            self.epsilon = max(self.epsilon_min, self.epsilon * adjusted_decay)
        else:
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

            self.decay_epsilon()

            if episode % 1000 == 0:
                print(
                    f"Episode {episode}, Epsilon: {self.epsilon:.4f}, Total Reward: {total_reward:.2f}"
                )

    def evaluate(self, env, episodes):
        total_rewards = 0
        total_wins = 0
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
            if episode_reward > 0:
                total_wins += 1

        self.epsilon = original_epsilon

        final_win_rate = (total_wins / episodes) * 100
        avg_reward = total_rewards / episodes
        print(f"Final Win Rate: {final_win_rate:.2f}%, Avg Reward: {avg_reward:.3f}")
        return final_win_rate


# https://gymnasium.farama.org/introduction/train_agent/
class QLearningAgent:
    def __init__(
        self,
        action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.999,
        curriculum_stage=None,
    ):
        self.q_table = {}
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01
        self.curriculum_stage = curriculum_stage

    def set_curriculum_stage(self, curriculum_stage):
        self.curriculum_stage = curriculum_stage

        # Always reset epsilon for current stage (not just when advancing)
        if curriculum_stage and curriculum_stage.stage_id > 1:
            # Progressive epsilon reduction for tabular agent
            if curriculum_stage.stage_id == 2:
                self.epsilon = max(0.15, self.epsilon)
            elif curriculum_stage.stage_id == 3:
                self.epsilon = max(0.1, self.epsilon)
            elif curriculum_stage.stage_id == 4:
                self.epsilon = max(0.08, self.epsilon)
            elif curriculum_stage.stage_id >= 5:
                self.epsilon = max(0.05, self.epsilon)
            print(
                f"   Reset epsilon to {self.epsilon:.3f} for stage {curriculum_stage.stage_id}"
            )

    def _get_learning_state(self, state):
        if len(state) <= 6:
            return state
        else:
            return state[:6]

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            valid_actions = self._get_valid_actions(state)
            return random.choice(valid_actions) if valid_actions else 0

        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return 0

        learning_state = self._get_learning_state(state)
        return max(
            valid_actions, key=lambda a: self.q_table.get((learning_state, a), 0)
        )

    def _get_valid_actions(self, state):
        if len(state) == 3:
            base_actions = [0, 1]
        else:
            player_sum = state[0] if len(state) > 0 else 0
            dealer_up_card = state[1] if len(state) > 1 else 0
            has_usable_ace = state[2] if len(state) > 2 else False
            can_split = state[3] if len(state) > 3 else False
            can_double = state[4] if len(state) > 4 else False
            is_blackjack = state[5] if len(state) > 5 else False
            can_surrender = state[6] if len(state) > 6 else False
            can_insure = state[7] if len(state) > 7 else False

            base_actions = [0]

            if player_sum < 21 and not is_blackjack:
                base_actions.append(1)

            if can_double:
                base_actions.append(2)

            if can_split:
                base_actions.append(3)

            if can_surrender:
                base_actions.append(4)

            if can_insure:
                base_actions.append(5)

        if self.curriculum_stage is not None:
            valid_actions = [
                a for a in base_actions if a in self.curriculum_stage.available_actions
            ]
            return valid_actions if valid_actions else [0]
        else:
            return base_actions

    def update(self, state, action, reward, next_state):
        learning_state = self._get_learning_state(state)
        learning_next_state = self._get_learning_state(next_state)

        old_value = self.q_table.get((learning_state, action), 0)

        valid_next_actions = self._get_valid_actions(learning_next_state)
        if valid_next_actions:
            next_max = max(
                [
                    self.q_table.get((learning_next_state, a), 0)
                    for a in valid_next_actions
                ]
            )
        else:
            next_max = 0

        new_value = (1 - self.lr) * old_value + self.lr * (
            reward + self.gamma * next_max
        )
        self.q_table[(learning_state, action)] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def evaluate(self, env, episodes):
        total_rewards = 0
        total_wins = 0
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
            if episode_reward > 0:
                total_wins += 1

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
