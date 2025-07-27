import os
import json
import time
from datetime import datetime
import numpy as np
from RLAgent import DQNAgent, QLearningAgent
from BlackJackENV import BlackjackEnv


class MultiAgentStandardSystem:
    def __init__(
        self,
        num_agents=3,
        agent_types=None,
        deck_type="infinite",
        penetration=0.9,
    ):
        self.num_agents = num_agents
        self.agent_types = agent_types or ["dqn", "tabular", "dqn"]
        self.deck_type = deck_type
        self.penetration = penetration
        self.setup_logging_directory(deck_type, penetration)
        self.agents = []
        for i, agent_type in enumerate(self.agent_types):
            if agent_type == "dqn":
                agent = DQNAgent(
                    action_space=[0, 1, 2, 3, 4, 5],
                    learning_rate=0.001,
                    exploration_rate=1.0,
                    exploration_decay=0.9999,
                    memory_size=50000,
                    batch_size=64,
                )
            else:
                agent = QLearningAgent(
                    action_space=[0, 1, 2, 3, 4, 5],
                    learning_rate=0.1,
                    exploration_rate=1.0,
                    exploration_decay=0.9999,
                )
            agent.agent_id = i
            agent.agent_type = agent_type
            self.agents.append(agent)
        self.global_performance_log = []

    def setup_logging_directory(self, deck_type, penetration):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        date_str = datetime.now().strftime("%Y%m%d")
        self.log_dir = (
            f"logs/logs-{date_str}-standard-{deck_type}-{penetration}-no-curriculum"
        )
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.eval_log_dir = os.path.join(self.log_dir, "evaluation")
        self.training_log_dir = os.path.join(self.log_dir, "training")
        self.report_log_dir = os.path.join(self.log_dir, "reports")
        for subdir in [self.eval_log_dir, self.training_log_dir, self.report_log_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        print(f"📁 Logging directory setup: {self.log_dir}")

    def train(self, total_episodes=50000, eval_episodes=1000):
        start_time = time.time()
        print(f"\n🤖 STANDARD MULTI-AGENT RL TRAINING")
        print("=" * 60)
        print(f"Agents: {self.num_agents} ({', '.join(self.agent_types)})")
        print(f"Total Episodes: {total_episodes}")

        for agent_idx, agent in enumerate(self.agents):
            print(f"\nTraining Agent {agent_idx} ({agent.agent_type.upper()})")
            env = BlackjackEnv(
                deck_type=self.deck_type,
                penetration=self.penetration,
            )
            episode_rewards = []
            wins = 0
            every_n_episodes_to_log = max(1, total_episodes // 100)
            for episode in range(total_episodes):
                state = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    if hasattr(agent, "remember"):
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay()
                    else:
                        agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                episode_rewards.append(total_reward)
                detailed_stats = env.get_detailed_win_stats()
                if detailed_stats:
                    episode_wins = 0
                    episode_losses = 0
                    for hand_detail in detailed_stats["hand_details"]:
                        bet_multiplier = 2 if hand_detail["doubled"] else 1
                        if hand_detail["result"] in ("win", "blackjack"):
                            episode_wins += bet_multiplier
                        elif hand_detail["result"] in ("lose", "bust"):
                            episode_losses += bet_multiplier
                    wins += episode_wins
                agent.decay_epsilon()
                if episode % every_n_episodes_to_log == 0:
                    print(
                        f"  Episode {episode}: Win Rate: {(wins/(episode+1))*100:.1f}%, Epsilon: {agent.epsilon:.4f}"
                    )
            # Save agent model
            models_dir = os.path.join(self.log_dir, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            filename = f"standard_agent_{agent.agent_type}_{agent_idx}"
            if agent.agent_type == "dqn":
                model_path = os.path.join(models_dir, f"{filename}.pth")
                agent.save_model(model_path)
            else:
                model_path = os.path.join(models_dir, f"{filename}.pkl")
                agent.save_model(model_path)
            print(f"Saved {filename} to {model_path}")
            # Evaluate agent
            eval_results = self.evaluate(agent, env, eval_episodes)
            print(
                f"  Final Win Rate: {eval_results['win_rate']*100:.2f}% | Avg Reward: {eval_results['avg_reward']:.2f}"
            )
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken: {time_taken:.2f} seconds")
            # Save evaluation log with enhanced statistics
            evaluation_log = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "evaluation_episodes": eval_episodes,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "win_rate": eval_results["win_rate"],
                    "avg_reward": eval_results["avg_reward"],
                    "action_performance": eval_results["action_performance"],
                    "game_outcomes": eval_results["game_outcomes"],
                    "game_outcome_percentages": eval_results[
                        "game_outcome_percentages"
                    ],
                    "strategy_table": eval_results["strategy_table"],
                    "state_action_stats": eval_results["state_action_stats"],
                    "state_win_stats": eval_results["state_win_stats"],
                    "state_reward_stats": eval_results["state_reward_stats"],
                    "time_taken": time_taken,
                },
            }

            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.eval_log_dir,
                f"evaluation_log_agent_{agent.agent_id}_{agent.agent_type}_{timestamp}.json",
            )
            with open(filename, "w") as f:
                json.dump(evaluation_log, f, indent=2)

            print(f"  📊 Evaluation log saved to: {filename}")
        print("\n✅ STANDARD TRAINING COMPLETE!")
        print(f"📁 All logs and models saved to: {self.log_dir}")

    def evaluate(self, agent, env, episodes):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        total_rewards = []
        wins = 0
        action_rewards = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        # Enhanced statistics tracking
        state_action_stats = {}  # Track actions per state
        state_win_stats = {}  # Track wins per state
        state_reward_stats = {}  # Track rewards per state
        game_outcomes = {
            "wins": 0,
            "losses": 0,
            "busts": 0,
            "pushes": 0,
            "blackjacks": 0,
        }

        for _ in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_actions = []
            episode_wins = 0

            while not done:
                action = agent.get_action(state)
                episode_actions.append(action)

                # Create state key for statistics
                player_sum = state[0]
                dealer_up = state[1]
                has_ace = state[2]
                state_key = f"P{player_sum}_D{dealer_up}_A{has_ace}"

                # Track state-action statistics
                if state_key not in state_action_stats:
                    state_action_stats[state_key] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                state_action_stats[state_key][action] += 1

                # Track state-reward statistics
                if state_key not in state_reward_stats:
                    state_reward_stats[state_key] = []
                if state_key not in state_win_stats:
                    state_win_stats[state_key] = {"wins": 0, "total": 0}

                state, reward, done = env.step(action)
                episode_reward += reward

                # Track state rewards
                state_reward_stats[state_key].append(reward)

            total_rewards.append(episode_reward)

            # Track action performance
            for action in episode_actions:
                action_rewards[action].append(episode_reward)

            detailed_stats = env.get_detailed_win_stats()
            if detailed_stats:
                episode_wins = 0
                episode_losses = 0
                for hand_detail in detailed_stats["hand_details"]:
                    bet_multiplier = 2 if hand_detail["doubled"] else 1
                    if hand_detail["result"] in ("win", "blackjack"):
                        episode_wins += bet_multiplier
                    elif hand_detail["result"] in ("lose", "bust"):
                        episode_losses += bet_multiplier
                wins += episode_wins

                # Track game outcomes
                for hand_detail in detailed_stats["hand_details"]:
                    result = hand_detail["result"]
                    if result == "win":
                        game_outcomes["wins"] += 1
                    elif result == "lose":
                        game_outcomes["losses"] += 1
                    elif result == "blackjack":
                        game_outcomes["blackjacks"] += 1
                    elif result == "push":
                        game_outcomes["pushes"] += 1
                    elif result == "bust":
                        game_outcomes["busts"] += 1

                # Track state win statistics
                for state_key in state_win_stats:
                    state_win_stats[state_key]["total"] += 1
                    if episode_wins > 0:
                        state_win_stats[state_key]["wins"] += 1

        agent.epsilon = original_epsilon

        # Calculate game outcome percentages
        total_hands = sum(game_outcomes.values())
        game_outcome_percentages = {}
        if total_hands > 0:
            game_outcome_percentages = {
                "win_percent": (game_outcomes["wins"] / total_hands) * 100,
                "lose_percent": (game_outcomes["losses"] / total_hands) * 100,
                "bust_percent": (game_outcomes["busts"] / total_hands) * 100,
                "push_percent": (game_outcomes["pushes"] / total_hands) * 100,
                "blackjack_percent": (game_outcomes["blackjacks"] / total_hands) * 100,
                "win_loss_ratio": (
                    (
                        (game_outcomes["wins"] + game_outcomes["blackjacks"])
                        / (game_outcomes["losses"] + game_outcomes["busts"])
                    )
                    if (game_outcomes["losses"] + game_outcomes["busts"]) > 0
                    else float("inf")
                ),
            }

        # Create strategy table data
        strategy_table = {}
        for state_key, action_counts in state_action_stats.items():
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                strategy_table[state_key] = {
                    "stand_percent": (action_counts[0] / total_actions) * 100,
                    "hit_percent": (action_counts[1] / total_actions) * 100,
                    "double_percent": (action_counts[2] / total_actions) * 100,
                    "split_percent": (action_counts[3] / total_actions) * 100,
                    "total_actions": total_actions,
                    "win_rate": (
                        (
                            state_win_stats[state_key]["wins"]
                            / state_win_stats[state_key]["total"]
                        )
                        * 100
                        if state_win_stats[state_key]["total"] > 0
                        else 0
                    ),
                    "avg_reward": (
                        np.mean(state_reward_stats[state_key])
                        if state_reward_stats[state_key]
                        else 0
                    ),
                }

        # Add action performance to return
        return {
            "win_rate": wins / episodes,
            "avg_reward": np.mean(total_rewards),
            "action_performance": {
                action: {
                    "count": len(rewards),
                    "avg_reward": np.mean(rewards) if rewards else 0,
                    "std_reward": np.std(rewards) if rewards else 0,
                }
                for action, rewards in action_rewards.items()
            },
            "game_outcomes": game_outcomes,
            "game_outcome_percentages": game_outcome_percentages,
            "strategy_table": strategy_table,
            "state_action_stats": state_action_stats,
            "state_win_stats": state_win_stats,
            "state_reward_stats": {
                k: {"avg": np.mean(v), "std": np.std(v), "count": len(v)}
                for k, v in state_reward_stats.items()
                if v
            },
        }
