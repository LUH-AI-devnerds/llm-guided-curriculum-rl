import numpy as np
import random
from collections import Counter


class BlackjackEnv:
    """
    An enhanced Blackjack environment for Reinforcement Learning with full rules.
    Supports both infinite deck and finite deck (shoe) options.

    Actions: 0=stand, 1=hit, 2=double_down, 3=split
    Deck Options: infinite, 1-deck, 6-deck, 8-deck (casino standard)
    """

    def __init__(
        self, curriculum_stage=3, deck_type="infinite", penetration=0.75, budget=100
    ):
        """
        Initialize the Blackjack environment.

        Args:
            curriculum_stage (int): Curriculum stage for action constraints
            deck_type (str): "infinite", "1-deck", "6-deck", or "8-deck"
            penetration (float): When to reshuffle (0.75 = reshuffle at 75% through deck)
            budget (int): Starting budget for the agent (default: 100)
        """
        self.curriculum_stage = curriculum_stage
        self.action_space = [0, 1, 2, 3]  # 0: stand, 1: hit, 2: double down, 3: split
        self.initial_bet = 1
        self.deck_type = deck_type
        self.penetration = penetration
        self.initial_budget = budget
        self.budget = budget
        self.total_winnings = 0
        self.total_losses = 0
        self.games_played = 0

        # Initialize deck
        self._initialize_deck()
        self.reset()

    def _initialize_deck(self):
        """Initialize the deck based on deck_type."""
        if self.deck_type == "infinite":
            self.deck = None
            self.cards_remaining = None
            self.total_cards = None
            self.shuffle_point = None
        else:
            # Parse deck type
            if self.deck_type == "1-deck":
                num_decks = 1
            elif self.deck_type == "6-deck":
                num_decks = 6
            elif self.deck_type == "8-deck":
                num_decks = 8
            else:
                raise ValueError(
                    f"Invalid deck_type: {self.deck_type}. Use 'infinite', '1-deck', '6-deck', or '8-deck'"
                )

            # Create standard 52-card deck
            standard_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4  # 4 suits
            self.deck = standard_deck * num_decks
            self.total_cards = len(self.deck)
            self.shuffle_point = int(self.total_cards * self.penetration)
            self._shuffle_deck()

    def _shuffle_deck(self):
        """Shuffle the deck and reset card counting."""
        if self.deck is not None:
            random.shuffle(self.deck)
            self.cards_remaining = self.deck.copy()
            self.card_counts = Counter(self.cards_remaining)
            # print(f"🃏 Shuffled {self.deck_type} deck ({len(self.deck)} cards)")

    def _draw_card(self):
        """Draw a single card from the deck."""
        if self.deck_type == "infinite":
            # Infinite deck: random card every time
            return random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
        else:
            # Finite deck: draw from remaining cards
            if len(self.cards_remaining) <= self.shuffle_point:
                # print(
                #     f"🔄 Reshuffling {self.deck_type} deck at {len(self.cards_remaining)} cards remaining"
                # )
                self._shuffle_deck()

            if not self.cards_remaining:
                self._shuffle_deck()

            card = self.cards_remaining.pop()
            self.card_counts[card] -= 1
            return card

    def get_card_counting_info(self):
        """Get card counting information for finite decks."""
        if self.deck_type == "infinite":
            return {
                "deck_type": "infinite",
                "cards_remaining": None,
                "penetration": None,
                "running_count": None,
                "true_count": None,
            }

        # Calculate running count (Hi-Lo system)
        running_count = 0
        for card, count in self.card_counts.items():
            if card in [2, 3, 4, 5, 6]:
                running_count += count  # Low cards: +1
            elif card in [10, 11]:
                running_count -= count  # High cards: -1
            # 7, 8, 9 are neutral (0)

        # Calculate true count (running count per deck remaining)
        decks_remaining = len(self.cards_remaining) / 52
        true_count = running_count / decks_remaining if decks_remaining > 0 else 0

        return {
            "deck_type": self.deck_type,
            "cards_remaining": len(self.cards_remaining),
            "penetration": len(self.cards_remaining) / self.total_cards,
            "running_count": running_count,
            "true_count": true_count,
            "card_distribution": dict(self.card_counts),
        }

    def reset(self):
        """Resets the game to a new hand."""
        self.player_hands = [
            [self._draw_card(), self._draw_card()]
        ]  # Support multiple hands for splits
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.current_hand_idx = 0
        self.bet_amounts = [self.initial_bet]
        self.doubled_down = [False]
        self.can_split = self._can_split()
        self.can_double = True
        self.game_over = False
        self.games_played += 1
        return self._get_state()

    def _can_split(self):
        """Check if current hand can be split."""
        if self.current_hand_idx >= len(self.player_hands):
            return False
        current_hand = self.player_hands[self.current_hand_idx]
        if len(current_hand) != 2:
            return False

        # Get card values for comparison (treat all 10-value cards as same)
        card1_val = 10 if current_hand[0] == 10 else current_hand[0]
        card2_val = 10 if current_hand[1] == 10 else current_hand[1]
        return card1_val == card2_val

    def _is_valid_action(self, action):
        """Check if action is valid in current state."""
        current_hand = self.player_hands[self.current_hand_idx]

        if action == 0:  # Stand - always valid
            return True
        elif action == 1:  # Hit - valid if not bust
            return self._get_hand_sum(current_hand) < 21
        elif action == 2:  # Double down - only with 2 cards
            return len(current_hand) == 2 and self.can_double
        elif action == 3:  # Split - only with matching pair
            return self._can_split() and len(self.player_hands) < 4
        return False

    def _get_hand_sum(self, hand):
        """Calculates the sum of a hand, handling Aces."""
        hand_sum = sum(hand)
        num_aces = hand.count(11)
        while hand_sum > 21 and num_aces:
            hand_sum -= 10
            num_aces -= 1
        return hand_sum

    def _get_state(self):
        """Gets the current state of the game."""
        # Handle case when game is over
        if self.game_over or self.current_hand_idx >= len(self.player_hands):
            # Return last valid hand state
            if self.player_hands:
                last_hand = self.player_hands[-1]
                player_sum = self._get_hand_sum(last_hand)
                dealer_up_card = self.dealer_hand[0]
                has_usable_ace = 11 in last_hand and self._get_hand_sum(last_hand) <= 21
                return (
                    player_sum,
                    dealer_up_card,
                    has_usable_ace,
                    False,
                    False,
                    False,
                    self.budget,
                    self.games_played,
                )
            else:
                return (
                    0,
                    0,
                    False,
                    False,
                    False,
                    False,
                    self.budget,
                    self.games_played,
                )

        current_hand = self.player_hands[self.current_hand_idx]
        player_sum = self._get_hand_sum(current_hand)
        dealer_up_card = self.dealer_hand[0]
        has_usable_ace = 11 in current_hand and self._get_hand_sum(current_hand) <= 21

        # Enhanced state representation
        can_split = (
            self._can_split() and len(self.player_hands) < 4
        )  # Limit to 4 hands max
        can_double = self.can_double and len(current_hand) == 2
        is_blackjack = len(current_hand) == 2 and player_sum == 21

        return (
            player_sum,
            dealer_up_card,
            has_usable_ace,
            can_split,
            can_double,
            is_blackjack,
            self.budget,
            self.games_played,
        )

    def step(self, action):
        """Performs an action and returns the next state, reward, and done flag."""
        if self.game_over:
            return self._get_state(), 0, True

        # Validate action
        if not self._is_valid_action(action):
            # Invalid action gets small penalty and no state change
            return self._get_state(), -0.1, False

        current_hand = self.player_hands[self.current_hand_idx]

        if action == 1:  # Hit
            current_hand.append(self._draw_card())
            player_sum = self._get_hand_sum(current_hand)
            self.can_double = False  # Can't double after hitting

            if player_sum > 21:  # Bust
                return self._move_to_next_hand()
            return self._get_state(), 0, False

        elif action == 2:  # Double Down
            if not (len(current_hand) == 2 and self.can_double):
                return self._get_state(), -0.1, False  # Invalid action

            self.bet_amounts[self.current_hand_idx] *= 2
            current_hand.append(self._draw_card())
            self.doubled_down[self.current_hand_idx] = True
            return self._move_to_next_hand()

        elif action == 3:  # Split
            if not self._can_split():
                return self._get_state(), -0.1, False  # Invalid action

            # Create new hand with second card
            card_to_split = current_hand.pop()
            new_hand = [card_to_split, self._draw_card()]
            current_hand.append(self._draw_card())

            self.player_hands.append(new_hand)
            self.bet_amounts.append(self.initial_bet)
            self.doubled_down.append(False)

            return self._get_state(), 0, False

        else:  # Stand (action == 0)
            return self._move_to_next_hand()

    def _move_to_next_hand(self):
        """Move to next hand or end player turn."""
        self.current_hand_idx += 1

        if self.current_hand_idx >= len(self.player_hands):
            # All hands played, dealer's turn
            return self._play_dealer_and_calculate_rewards()
        else:
            # Move to next hand
            self.can_double = True
            return self._get_state(), 0, False

    def _calculate_dynamic_reward(self, base_reward, bet_amount):
        """
        Calculate dynamic reward based on budget status and performance.

        Args:
            base_reward (float): Base reward from game outcome
            bet_amount (float): Amount bet on this hand

        Returns:
            float: Scaled reward based on budget and performance
        """
        # Budget-based scaling factors
        budget_ratio = self.budget / self.initial_budget

        # Performance-based scaling
        if self.games_played > 1:
            win_rate = (
                self.total_winnings / (self.total_winnings + self.total_losses)
                if (self.total_winnings + self.total_losses) > 0
                else 0.5
            )
        else:
            win_rate = 0.5

        # Dynamic reward scaling
        if base_reward > 0:  # Winning
            # Higher rewards when budget is low (comeback bonus)
            budget_bonus = max(1.0, (1.0 - budget_ratio) * 2.0)
            # Higher rewards for consistent winning
            performance_bonus = 1.0 + (win_rate - 0.5) * 0.5
            scaled_reward = base_reward * budget_bonus * performance_bonus

        elif base_reward < 0:  # Losing
            # Higher penalties when budget is low (risk of going broke)
            budget_penalty = max(1.0, (1.0 - budget_ratio) * 3.0)
            # Higher penalties for poor performance
            performance_penalty = 1.0 + (0.5 - win_rate) * 0.5
            scaled_reward = base_reward * budget_penalty * performance_penalty

        else:  # Push
            scaled_reward = 0.0

        # Severe penalty for going broke
        if self.budget <= 0:
            scaled_reward -= 50.0  # Heavy penalty for bankruptcy

        return scaled_reward

    def _play_dealer_and_calculate_rewards(self):
        """Play dealer's turn and calculate final rewards."""
        dealer_sum = self._get_hand_sum(self.dealer_hand)

        # Dealer draws cards
        while dealer_sum < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_sum = self._get_hand_sum(self.dealer_hand)

        # Calculate total reward across all hands
        total_reward = 0
        dealer_busted = dealer_sum > 21
        dealer_blackjack = len(self.dealer_hand) == 2 and dealer_sum == 21

        for i, hand in enumerate(self.player_hands):
            player_sum = self._get_hand_sum(hand)
            bet = self.bet_amounts[i]

            if player_sum > 21:  # Player busted
                hand_reward = -bet
                self.total_losses += bet
                self.budget -= bet
            elif (
                len(hand) == 2 and player_sum == 21 and not dealer_blackjack
            ):  # Player blackjack
                hand_reward = bet * 1.5  # Blackjack pays 3:2
                self.total_winnings += hand_reward
                self.budget += hand_reward
            elif dealer_busted or player_sum > dealer_sum:
                hand_reward = bet
                self.total_winnings += hand_reward
                self.budget += hand_reward
            elif player_sum < dealer_sum:
                hand_reward = -bet
                self.total_losses += bet
                self.budget -= bet
            else:  # Push
                hand_reward = 0

            # Apply dynamic reward scaling
            scaled_reward = self._calculate_dynamic_reward(hand_reward, bet)
            total_reward += scaled_reward

        self.game_over = True
        return self._get_state(), total_reward, True

    def get_game_info(self):
        """Get detailed information about the current game state."""
        info = {
            "deck_info": self.get_card_counting_info(),
            "player_hands": self.player_hands,
            "dealer_hand": self.dealer_hand,
            "current_hand_idx": self.current_hand_idx,
            "bet_amounts": self.bet_amounts,
            "game_over": self.game_over,
            "budget": self.budget,
            "initial_budget": self.initial_budget,
            "total_winnings": self.total_winnings,
            "total_losses": self.total_losses,
            "games_played": self.games_played,
            "budget_ratio": (
                self.budget / self.initial_budget if self.initial_budget > 0 else 0
            ),
        }
        return info

    def get_detailed_win_stats(self):
        """Get detailed win statistics that properly account for double down and split scenarios."""
        if not self.game_over:
            return None

        stats = {
            "total_hands": len(self.player_hands),
            "total_bet": sum(self.bet_amounts),
            "hands_won": 0,
            "hands_lost": 0,
            "hands_pushed": 0,
            "double_downs": sum(self.doubled_down),
            "splits": len(self.player_hands) - 1,  # Original hand + splits
            "blackjacks": 0,
            "busts": 0,
            "hand_details": [],
        }

        dealer_sum = self._get_hand_sum(self.dealer_hand)
        dealer_busted = dealer_sum > 21
        dealer_blackjack = len(self.dealer_hand) == 2 and dealer_sum == 21

        for i, hand in enumerate(self.player_hands):
            player_sum = self._get_hand_sum(hand)
            bet = self.bet_amounts[i]
            is_doubled = self.doubled_down[i]

            hand_detail = {
                "hand_index": i,
                "cards": hand.copy(),
                "sum": player_sum,
                "bet": bet,
                "doubled": is_doubled,
                "result": None,
                "reward": 0,
            }

            # Determine hand result
            if player_sum > 21:  # Bust
                hand_detail["result"] = "bust"
                hand_detail["reward"] = -bet
                stats["busts"] += 1
                stats["hands_lost"] += 1
            elif (
                len(hand) == 2 and player_sum == 21 and not dealer_blackjack
            ):  # Blackjack
                hand_detail["result"] = "blackjack"
                hand_detail["reward"] = bet * 1.5
                stats["blackjacks"] += 1
                stats["hands_won"] += 1
            elif dealer_busted or player_sum > dealer_sum:  # Win
                hand_detail["result"] = "win"
                hand_detail["reward"] = bet
                stats["hands_won"] += 1
            elif player_sum < dealer_sum:  # Loss
                hand_detail["result"] = "loss"
                hand_detail["reward"] = -bet
                stats["hands_lost"] += 1
            else:  # Push
                hand_detail["result"] = "push"
                hand_detail["reward"] = 0
                stats["hands_pushed"] += 1

            stats["hand_details"].append(hand_detail)

        # Calculate win rate considering bet amounts
        total_wagered = sum(self.bet_amounts)
        total_won = sum(
            detail["reward"] for detail in stats["hand_details"] if detail["reward"] > 0
        )
        total_lost = abs(
            sum(
                detail["reward"]
                for detail in stats["hand_details"]
                if detail["reward"] < 0
            )
        )

        stats["total_won"] = total_won
        stats["total_lost"] = total_lost
        stats["net_result"] = total_won - total_lost
        stats["win_rate_by_hands"] = (
            stats["hands_won"] / stats["total_hands"] if stats["total_hands"] > 0 else 0
        )
        stats["win_rate_by_bets"] = (
            total_won / total_wagered if total_wagered > 0 else 0
        )

        return stats


# Enhanced BlackjackEnv with finite deck support and card counting
# Maintains backward compatibility with original BlackjackEnv interface


def demo_finite_deck():
    """Demonstrate the finite deck functionality."""
    print("🎲 FINITE DECK BLACKJACK DEMO")
    print("=" * 50)

    # Test different deck types
    deck_types = ["infinite", "1-deck", "6-deck", "8-deck"]

    for deck_type in deck_types:
        print(f"\n🃏 Testing {deck_type.upper()} deck:")
        print("-" * 30)

        env = BlackjackEnv(deck_type=deck_type, penetration=0.75)

        # Play a few hands
        for hand in range(3):
            state = env.reset()
            deck_info = env.get_card_counting_info()

            print(f"Hand {hand + 1}:")
            print(f"  Player: {env.player_hands[0]}")
            print(f"  Dealer: [{env.dealer_hand[0]}, ?]")
            print(f"  Cards remaining: {deck_info['cards_remaining']}")
            if deck_info["running_count"] is not None:
                print(f"  Running count: {deck_info['running_count']}")
                print(f"  True count: {deck_info['true_count']:.2f}")

            # Play one action (hit)
            state, reward, done = env.step(1)  # Hit
            print(f"  After hit: {env.player_hands[0]}, Reward: {reward}")
            print()


if __name__ == "__main__":
    demo_finite_deck()
