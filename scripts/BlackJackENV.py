import numpy as np
import random


class BlackjackEnv:
    """
    An advanced Blackjack environment for Reinforcement Learning with full rules.
    Actions: 0=stand, 1=hit, 2=double_down, 3=split
    """

    def __init__(self, curriculum_stage=3):
        self.curriculum_stage = curriculum_stage
        self.action_space = [0, 1, 2, 3]  # 0: stand, 1: hit, 2: double down, 3: split
        self.initial_bet = 1
        self.reset()

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
        return self._get_state()

    def _draw_card(self):
        """Draws a single card from an infinite deck."""
        return random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

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
                return (player_sum, dealer_up_card, has_usable_ace, False, False, False)
            else:
                return (0, 0, False, False, False, False)

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
            elif (
                len(hand) == 2 and player_sum == 21 and not dealer_blackjack
            ):  # Player blackjack
                hand_reward = bet * 1.5  # Blackjack pays 3:2
            elif dealer_busted or player_sum > dealer_sum:
                hand_reward = bet
            elif player_sum < dealer_sum:
                hand_reward = -bet
            else:  # Push
                hand_reward = 0

            total_reward += hand_reward

        self.game_over = True
        return self._get_state(), total_reward, True
