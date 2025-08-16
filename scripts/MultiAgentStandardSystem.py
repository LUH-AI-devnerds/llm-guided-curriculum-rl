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
                    learning_rate=0.0005,
                    exploration_rate=1.0,
                    exploration_decay=0.99995,
                    memory_size=100000,
                    batch_size=128,
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

        # Track action usage to detect degenerate behavior
        self.action_usage_history = {i: [] for i in range(6)}
        self.episode_count = 0

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
        print(f"Logging directory setup: {self.log_dir}")

    def _apply_action_masking(
        self, action, state, episode_actions, episode, total_episodes
    ):
        """
        Apply intelligent action masking to prevent degenerate behavior
        """
        player_sum, dealer_up, has_ace = state[0], state[1], state[2]

        # Progressive action availability based on training progress
        training_progress = episode / total_episodes

        # Early training: restrict to basic actions
        if training_progress < 0.3:
            if action in [4, 5]:  # Surrender, Insurance
                # Force to basic strategy action instead
                if player_sum >= 17:
                    return 0  # Stand
                elif player_sum <= 11:
                    return 1  # Hit
                else:
                    return 0 if player_sum >= 12 and dealer_up <= 6 else 1

        # Mid training: gradually introduce advanced actions
        elif training_progress < 0.6:
            # Limit surrender usage
            if action == 4:
                surrender_count = episode_actions.count(4)
                if surrender_count >= 1:  # Max 1 surrender per episode
                    return 0  # Force stand instead
                # Only allow surrender in truly bad situations
                if not (player_sum >= 15 and dealer_up in [9, 10, 11]):
                    return 0

        # Late training: allow all actions but with constraints
        else:
            # Prevent excessive surrender usage
            if action == 4:
                recent_actions = (
                    episode_actions[-5:]
                    if len(episode_actions) >= 5
                    else episode_actions
                )
                surrender_ratio = recent_actions.count(4) / max(len(recent_actions), 1)
                if (
                    surrender_ratio > 0.3
                ):  # Don't surrender more than 30% of recent actions
                    return 0

        # Check for action loops
        if len(episode_actions) >= 3:
            if all(a == action for a in episode_actions[-3:]):
                # Break the loop with a different action
                if action == 4:  # If stuck on surrender
                    return 0 if player_sum >= 17 else 1
                elif action == 1:  # If stuck on hit
                    return 0
                elif action == 0:  # If stuck on stand
                    return 1 if player_sum < 17 else 0

        return action

    def _apply_reward_shaping(
        self, reward, action, state, episode_actions, episode, total_episodes
    ):
        """
        Apply reward shaping to guide learning away from degenerate policies
        """
        shaped_reward = reward
        player_sum = state[0]

        # Penalty for excessive surrender
        if action == 4:
            surrender_count = episode_actions.count(4)
            if surrender_count > 1:
                shaped_reward -= 1.0  # Additional penalty for multiple surrenders

            # Penalty for surrendering good hands
            if player_sum <= 14:
                shaped_reward -= 0.5

        # Encourage exploration in early training
        training_progress = episode / total_episodes
        if training_progress < 0.5:
            # Small bonus for using different actions
            unique_actions = len(set(episode_actions))
            if unique_actions >= 2:
                shaped_reward += 0.1

        # Penalty for action repetition
        if len(episode_actions) >= 4:
            recent_actions = episode_actions[-4:]
            if len(set(recent_actions)) == 1:  # All same action
                shaped_reward -= 0.5

        return shaped_reward

    def train(self, total_episodes=50000, eval_episodes=1000):
        start_time = time.time()
        print(f"\nSTANDARD MULTI-AGENT RL TRAINING")
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
                step_count = 0
                max_steps_per_episode = 50
                consecutive_same_action = 0
                last_action = None
                episode_actions = []

                while not done and step_count < max_steps_per_episode:
                    action = agent.get_action(state)

                    # Apply action masking to prevent degenerate behavior
                    action = self._apply_action_masking(
                        action, state, episode_actions, episode, total_episodes
                    )

                    episode_actions.append(action)

                    # Track consecutive same actions to detect loops
                    if action == last_action:
                        consecutive_same_action += 1
                    else:
                        consecutive_same_action = 0
                        last_action = action

                    # Force termination if stuck in action loop
                    if consecutive_same_action >= 5:
                        reward = -10.0  # Heavy penalty for getting stuck
                        done = True
                        break

                    next_state, reward, done = env.step(action)

                    # Apply reward shaping to discourage degenerate policies
                    reward = self._apply_reward_shaping(
                        reward, action, state, episode_actions, episode, total_episodes
                    )

                    if hasattr(agent, "remember"):
                        agent.remember(state, action, reward, next_state, done)
                        # Use same replay strategy as curriculum system
                        if episode < total_episodes // 2:
                            agent.replay()
                        else:
                            if episode % 3 == 0:
                                agent.replay()
                    else:
                        agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    step_count += 1

                if step_count >= max_steps_per_episode:
                    total_reward -= 5.0  # Penalty for exceeding max steps
                    print(
                        f"  [Train] Episode {episode} exceeded max steps ({max_steps_per_episode}), forcing termination. Actions: {episode_actions[-10:]}",
                        flush=True,
                    )
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

            print(f"  Evaluation log saved to: {filename}")
        print("\n STANDARD TRAINING COMPLETE!")
        print(f"All logs and models saved to: {self.log_dir}")

    def evaluate(self, agent, env, episodes, heavy_stats_threshold=20000):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

        # Automatically switch to lightweight evaluation for very large episode counts
        collect_heavy_stats = episodes <= heavy_stats_threshold

        total_rewards_sum = 0.0
        total_rewards = []  # kept only to compute std if heavy; otherwise unused
        wins = 0

        if collect_heavy_stats:
            action_rewards = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            state_action_stats = {}
            state_win_stats = {}
            state_reward_stats = {}
        else:
            # O(1) aggregations per action: count, sum, sum_sq
            action_aggs = {
                a: {"count": 0, "sum": 0.0, "sum_sq": 0.0} for a in [0, 1, 2, 3, 4, 5]
            }
            state_action_stats = {}
            state_win_stats = {}
            state_reward_stats = {}

        game_outcomes = {
            "wins": 0,
            "losses": 0,
            "busts": 0,
            "pushes": 0,
            "blackjacks": 0,
        }

        log_every = max(1, min(episodes // 20, 1000))

        print(f"  [Eval] Starting evaluation of {episodes} episodes...", flush=True)

        for episode_idx in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_actions = []
            episode_wins = 0
            visited_states = set() if collect_heavy_stats else None

            step_count = 0
            max_steps_per_episode = 50
            consecutive_same_action = 0
            last_action = None

            while not done and step_count < max_steps_per_episode:
                action = agent.get_action(state)

                # Apply same action masking for evaluation consistency
                action = self._apply_action_masking(
                    action, state, episode_actions, 1, 1
                )  # Use dummy values for evaluation

                episode_actions.append(action)

                # Track consecutive same actions to detect loops
                if action == last_action:
                    consecutive_same_action += 1
                else:
                    consecutive_same_action = 0
                    last_action = action

                # Force termination if stuck in action loop
                if consecutive_same_action >= 5:
                    episode_reward -= 10.0  # Heavy penalty for getting stuck
                    done = True
                    break

                # Create state key for statistics
                player_sum = state[0]
                dealer_up = state[1]
                has_ace = state[2]
                state_key = f"P{player_sum}_D{dealer_up}_A{has_ace}"

                if collect_heavy_stats:
                    # Track state-action statistics
                    if state_key not in state_action_stats:
                        state_action_stats[state_key] = {
                            0: 0,
                            1: 0,
                            2: 0,
                            3: 0,
                            4: 0,
                            5: 0,
                        }
                    state_action_stats[state_key][action] += 1

                    # Track state-reward statistics
                    if state_key not in state_reward_stats:
                        state_reward_stats[state_key] = []
                    if state_key not in state_win_stats:
                        state_win_stats[state_key] = {"wins": 0, "total": 0}
                    visited_states.add(state_key)

                state, reward, done = env.step(action)
                episode_reward += reward
                step_count += 1

                # Track state rewards only in heavy stats mode
                if collect_heavy_stats:
                    state_reward_stats[state_key].append(reward)

            if step_count >= max_steps_per_episode:
                episode_reward -= 5.0  # Penalty for exceeding max steps
                print(
                    f"  [Eval] Episode {episode_idx + 1} exceeded max steps ({max_steps_per_episode}), forcing termination. Actions: {episode_actions[-10:]}",
                    flush=True,
                )

            total_rewards_sum += episode_reward
            if collect_heavy_stats:
                total_rewards.append(episode_reward)
                # Track action performance (heavy mode keeps per-episode arrays)
                for action in episode_actions:
                    action_rewards[action].append(episode_reward)
            else:
                # Lightweight: aggregate per action
                for action in episode_actions:
                    agg = action_aggs[action]
                    agg["count"] += 1
                    agg["sum"] += episode_reward
                    agg["sum_sq"] += episode_reward * episode_reward

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

                if collect_heavy_stats:
                    # Track state win statistics only for states visited in this episode
                    for state_key in visited_states:
                        state_win_stats[state_key]["total"] += 1
                        if episode_wins > 0:
                            state_win_stats[state_key]["wins"] += 1

            # Periodic progress logging during evaluation to avoid appearing stuck
            if (episode_idx + 1) % log_every == 0 or episode_idx < 10:
                avg_reward_so_far = (
                    (total_rewards_sum / (episode_idx + 1))
                    if (episode_idx + 1) > 0
                    else 0.0
                )
                print(
                    f"  [Eval] Episode {episode_idx + 1}/{episodes} | Avg Reward: {avg_reward_so_far:.3f}",
                    flush=True,
                )

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
        if collect_heavy_stats:
            for state_key, action_counts in state_action_stats.items():
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    strategy_table[state_key] = {
                        "stand_percent": (action_counts[0] / total_actions) * 100,
                        "hit_percent": (action_counts[1] / total_actions) * 100,
                        "double_percent": (action_counts[2] / total_actions) * 100,
                        "split_percent": (action_counts[3] / total_actions) * 100,
                        "surrender_percent": (action_counts[4] / total_actions) * 100,
                        "insurance_percent": (action_counts[5] / total_actions) * 100,
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
        if collect_heavy_stats:
            action_performance = {
                action: {
                    "count": len(rewards),
                    "avg_reward": np.mean(rewards) if rewards else 0,
                    "std_reward": np.std(rewards) if rewards else 0,
                }
                for action, rewards in action_rewards.items()
            }
            avg_reward = np.mean(total_rewards) if total_rewards else 0.0
            state_reward_stats_summary = {
                k: {"avg": np.mean(v), "std": np.std(v), "count": len(v)}
                for k, v in state_reward_stats.items()
                if v
            }
        else:
            # Compute performance from aggregates
            action_performance = {}
            for a, agg in action_aggs.items():
                if agg["count"] > 0:
                    mean = agg["sum"] / agg["count"]
                    var = max(agg["sum_sq"] / agg["count"] - mean * mean, 0.0)
                    std = np.sqrt(var)
                else:
                    mean = 0.0
                    std = 0.0
                action_performance[a] = {
                    "count": agg["count"],
                    "avg_reward": mean,
                    "std_reward": std,
                }
            avg_reward = total_rewards_sum / episodes if episodes > 0 else 0.0
            state_reward_stats_summary = {}

        return {
            "win_rate": wins / episodes,
            "avg_reward": avg_reward,
            "action_performance": action_performance,
            "game_outcomes": game_outcomes,
            "game_outcome_percentages": game_outcome_percentages,
            "strategy_table": strategy_table if collect_heavy_stats else {},
            "state_action_stats": state_action_stats if collect_heavy_stats else {},
            "state_win_stats": state_win_stats if collect_heavy_stats else {},
            "state_reward_stats": state_reward_stats_summary,
        }
