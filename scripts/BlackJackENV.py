import numpy as np
import random
from collections import Counter


class BlackjackEnv:
    def __init__(
        self,
        curriculum_stage=3,
        deck_type="infinite",
        penetration=0.75,
    ):
        self.curriculum_stage = curriculum_stage
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.deck_type = deck_type
        self.penetration = penetration
        self.games_played = 0

        self._initialize_deck()
        self.reset()

    def _initialize_deck(self):
        if self.deck_type == "infinite":
            self.deck = None
            self.cards_remaining = None
            self.total_cards = None
            self.shuffle_point = None
        else:
            if self.deck_type == "1-deck":
                num_decks = 1
            elif self.deck_type == "4-deck":
                num_decks = 4
            elif self.deck_type == "8-deck":
                num_decks = 8
            else:
                raise ValueError(
                    f"Invalid deck_type: {self.deck_type}. Use 'infinite', '1-deck', '6-deck', or '8-deck'"
                )

            standard_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
            self.deck = standard_deck * num_decks
            self.total_cards = len(self.deck)
            self.shuffle_point = int(self.total_cards * self.penetration)
            self._shuffle_deck()

    def _shuffle_deck(self):
        if self.deck is not None:
            random.shuffle(self.deck)
            self.cards_remaining = self.deck.copy()
            self.card_counts = Counter(self.cards_remaining)

    def _draw_card(self):
        if self.deck_type == "infinite":
            return random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
        else:
            if len(self.cards_remaining) <= self.shuffle_point:
                self._shuffle_deck()

            if not self.cards_remaining:
                self._shuffle_deck()

            card = self.cards_remaining.pop()
            self.card_counts[card] -= 1
            return card

    def get_card_counting_info(self):
        if self.deck_type == "infinite":
            return {
                "deck_type": "infinite",
                "cards_remaining": None,
                "penetration": None,
                "running_count": None,
                "true_count": None,
            }

        running_count = 0
        for card, count in self.card_counts.items():
            if card in [2, 3, 4, 5, 6]:
                running_count += count
            elif card in [10, 11]:
                running_count -= count

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
        self.player_hands = [[self._draw_card(), self._draw_card()]]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.current_hand_idx = 0
        self.doubled_down = [False]
        self.surrendered_hands = [False]
        self.insurance_bets = [0]
        self.can_split = self._can_split()
        self.can_double = True
        self.can_surrender = True
        self.can_insure = True
        self.game_over = False
        self.games_played += 1
        return self._get_state()

    def _can_split(self):
        if self.current_hand_idx >= len(self.player_hands):
            return False
        current_hand = self.player_hands[self.current_hand_idx]
        if len(current_hand) != 2:
            return False

        card1_val = 10 if current_hand[0] == 10 else current_hand[0]
        card2_val = 10 if current_hand[1] == 10 else current_hand[1]
        return card1_val == card2_val

    def _dealer_has_blackjack(self):
        if len(self.dealer_hand) != 2:
            return False
        return self._get_hand_sum(self.dealer_hand) == 21

    def _is_valid_double_down(self, player_sum, dealer_up, has_usable_ace=False):
        """
        Check if double down is strategically valid based on basic strategy
        Handles both hard and soft hands
        """
        # Never allow double down on blackjack (21 with 2 cards)
        if player_sum == 21:
            return False

        # For soft hands (has usable ace), different rules apply
        if has_usable_ace:
            # Soft double down rules
            if player_sum >= 17:  # Soft 20+ (A-9, A-10, A-A)
                return False  # Never double soft 20+
            elif player_sum == 16:  # Soft 16 (A-5)
                return dealer_up in [4, 5, 6]  # Double soft 16 vs dealer 4-6
            elif player_sum == 15:  # Soft 15 (A-4)
                return dealer_up in [4, 5, 6]  # Double soft 15 vs dealer 4-6
            elif player_sum == 14:  # Soft 14 (A-3)
                return dealer_up in [5, 6]  # Double soft 14 vs dealer 5-6
            elif player_sum == 13:  # Soft 13 (A-2)
                return dealer_up in [5, 6]  # Double soft 13 vs dealer 5-6
            else:
                return False  # No valid soft doubles below 13

        # For hard hands (no usable ace)
        # Never double on 17, 18, 19, 20, or 21
        if player_sum >= 17:
            return False

        # Specific valid double down scenarios for hard hands
        valid_doubles = {
            8: [5, 6],  # Double 8 vs dealer 5-6
            9: [3, 4, 5, 6],  # Double 9 vs dealer 3-6
            10: [2, 3, 4, 5, 6, 7, 8, 9],  # Double 10 vs dealer 2-9
            11: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Double 11 vs any dealer
        }

        # Only allow double down if it's in the valid scenarios
        return player_sum in valid_doubles and dealer_up in valid_doubles[player_sum]

    def _is_valid_action(self, action):
        if self.game_over or self.current_hand_idx >= len(self.player_hands):
            return False

        current_hand = self.player_hands[self.current_hand_idx]
        player_sum = self._get_hand_sum(current_hand)
        dealer_up = self.dealer_hand[0]
        # Fix soft hand detection: check if ace is usable (not causing bust)
        # A soft hand has an ace that can be counted as 11 without busting
        has_usable_ace = (
            11 in current_hand
            and player_sum <= 21
            and player_sum >= 12
            and player_sum != 21
        )

        if action == 0:
            return True
        elif action == 1:
            return player_sum < 21
        elif action == 2:
            # Check both technical and strategic validity
            return (
                len(current_hand) == 2
                and self.can_double
                and self._is_valid_double_down(player_sum, dealer_up, has_usable_ace)
            )
        elif action == 3:
            return self._can_split() and len(self.player_hands) < 4
        elif action == 4:
            return (
                len(current_hand) == 2
                and self.can_surrender
                and not self._dealer_has_blackjack()
            )
        elif action == 5:
            return self.can_insure and self.dealer_hand[0] == 11
        return False

    def _get_hand_sum(self, hand):
        hand_sum = sum(hand)
        num_aces = hand.count(11)
        while hand_sum > 21 and num_aces:
            hand_sum -= 10
            num_aces -= 1
        return hand_sum

    def get_reward(
        self,
        agent_hand,
        dealer_hand,
        has_busted,
        action_taken=None,
        original_sum=None,
        dealer_up=None,
    ):
        agent_total = self._get_hand_sum(agent_hand)
        dealer_total = self._get_hand_sum(dealer_hand)

        if has_busted:
            return -1.0
        elif (
            len(agent_hand) == 2
            and agent_total == 21
            and not self._dealer_has_blackjack()
        ):
            return 1.5  # Blackjack pays 3:2
        elif dealer_total > 21:
            return 1.0  # Dealer bust
        elif agent_total > dealer_total:
            return 1.0  # Win
        elif agent_total < dealer_total:
            return -1.0  # Loss
        else:
            return 0.0  # Push

    def get_enhanced_reward(
        self,
        agent_hand,
        dealer_hand,
        has_busted,
        hand_index=0,
    ):
        """
        Enhanced reward function that considers strategic elements
        """
        agent_total = self._get_hand_sum(agent_hand)
        dealer_total = self._get_hand_sum(dealer_hand)

        if has_busted:
            if self.doubled_down[hand_index]:
                return -6.0
            else:
                return -2.0

        # Base reward with proper blackjack handling
        if (
            len(agent_hand) == 2
            and agent_total == 21
            and not self._dealer_has_blackjack()
        ):
            base_reward = 1.5  # Blackjack pays 3:2
        elif dealer_total > 21:
            base_reward = 1.0  # Dealer bust
            if self.doubled_down[hand_index]:
                base_reward *= 2.1
        elif agent_total > dealer_total:
            base_reward = 1.0  # Win
            if self.doubled_down[hand_index]:
                base_reward *= 2
        elif agent_total < dealer_total:
            base_reward = -1.0  # Loss
            if self.doubled_down[hand_index]:
                base_reward *= 2.1
        else:
            base_reward = 0.0  # Push

        # Strategic bonuses
        strategic_bonus = 0.0

        # 1. Double down strategic evaluation
        if self.doubled_down[hand_index] and hasattr(self, "double_down_info"):
            double_info = self.double_down_info
            if double_info["was_valid"]:
                strategic_bonus += 0.3  # Good strategic decision
            # No penalty needed since invalid doubles are blocked entirely

        # 2. Insurance evaluation
        if self.insurance_bets[hand_index] > 0:
            if self._dealer_has_blackjack():
                strategic_bonus += 0.2  # Successful insurance
            else:
                strategic_bonus -= 0.1  # Failed insurance

        # 3. Split evaluation
        if len(self.player_hands) > 1 and hand_index > 0:
            if agent_total <= 21 and (dealer_total > 21 or agent_total > dealer_total):
                strategic_bonus += 0.2  # Successful split

        return base_reward + strategic_bonus

    def _get_state(self):
        if self.game_over or self.current_hand_idx >= len(self.player_hands):
            if self.player_hands:
                last_hand = self.player_hands[-1]
                player_sum = self._get_hand_sum(last_hand)
                dealer_up_card = self.dealer_hand[0]
                has_usable_ace = (
                    11 in last_hand
                    and self._get_hand_sum(last_hand) <= 21
                    and self._get_hand_sum(last_hand) != 21
                )
                return (
                    player_sum,
                    dealer_up_card,
                    has_usable_ace,
                    False,
                    False,
                    False,
                )
            else:
                return (
                    0,
                    0,
                    False,
                    False,
                    False,
                    False,
                )

        current_hand = self.player_hands[self.current_hand_idx]
        player_sum = self._get_hand_sum(current_hand)
        dealer_up_card = self.dealer_hand[0]
        has_usable_ace = (
            11 in current_hand
            and self._get_hand_sum(current_hand) <= 21
            and self._get_hand_sum(current_hand) != 21
        )

        can_split = self._can_split() and len(self.player_hands) < 4
        can_double = (
            self.can_double
            and len(current_hand) == 2
            and self._is_valid_double_down(player_sum, dealer_up_card, has_usable_ace)
        )
        is_blackjack = len(current_hand) == 2 and player_sum == 21
        can_surrender = (
            self.can_surrender
            and len(current_hand) == 2
            and not self._dealer_has_blackjack()
        )
        can_insure = self.can_insure and self.dealer_hand[0] == 11

        card_count_info = self.get_card_counting_info()
        running_count = card_count_info.get("running_count", 0)
        true_count = card_count_info.get("true_count", 0)

        hand_type = 0
        if can_split:
            hand_type = 2
        elif has_usable_ace and player_sum <= 21:
            hand_type = 1

        return (
            player_sum,
            dealer_up_card,
            has_usable_ace,
            can_split,
            can_double,
            is_blackjack,
            can_surrender,
            can_insure,
            running_count,
            true_count,
            hand_type,
        )

    def step(self, action):
        if self.game_over:
            return self._get_state(), 0, True

        if not self._is_valid_action(action):
            return self._get_state(), -0.1, False

        current_hand = self.player_hands[self.current_hand_idx]

        if action == 1:
            current_hand.append(self._draw_card())
            player_sum = self._get_hand_sum(current_hand)
            self.can_double = False

            if player_sum > 21:
                return self._move_to_next_hand()
            return self._get_state(), 0, False

        elif action == 2:
            # NUCLEAR OPTION: COMPLETELY BLOCK invalid double downs
            original_sum = self._get_hand_sum(current_hand)
            dealer_up = self.dealer_hand[0]
            has_usable_ace = (
                11 in current_hand
                and original_sum <= 21
                and original_sum >= 12
                and original_sum != 21
            )

            # Check if this is a valid double down
            was_valid = self._is_valid_double_down(
                original_sum, dealer_up, has_usable_ace
            )

            # Store strategic decision info
            self.double_down_info = {
                "original_sum": original_sum,
                "dealer_up": dealer_up,
                "has_usable_ace": has_usable_ace,
                "was_valid": was_valid,
            }

            # NUCLEAR OPTION: COMPLETELY BLOCK invalid double downs
            if not was_valid:
                # Return massive penalty and DON'T execute the action
                return self._get_state(), -5.0, False

            # Valid double down - proceed normally
            current_hand.append(self._draw_card())
            self.doubled_down[self.current_hand_idx] = True

            next_state, final_reward, done = self._move_to_next_hand()
            return next_state, 0, done

        elif action == 3:
            if not self._can_split():
                return self._get_state(), -0.1, False

            card_to_split = current_hand.pop()
            new_hand = [card_to_split, self._draw_card()]
            current_hand.append(self._draw_card())

            self.player_hands.append(new_hand)
            self.doubled_down.append(False)
            self.surrendered_hands.append(False)
            self.insurance_bets.append(0)

            return self._get_state(), 0, False

        elif action == 4:
            if not (
                len(current_hand) == 2
                and self.can_surrender
                and not self._dealer_has_blackjack()
            ):
                return self._get_state(), -0.1, False

            self.surrendered_hands[self.current_hand_idx] = True
            return self._move_to_next_hand()

        elif action == 5:
            if not (self.can_insure and self.dealer_hand[0] == 11):
                return self._get_state(), -0.1, False

            self.insurance_bets[self.current_hand_idx] = 0.5
            self.can_insure = False

            # Insurance result will be evaluated in final reward
            return self._get_state(), 0, False

        else:
            return self._move_to_next_hand()

    def _move_to_next_hand(self):
        self.current_hand_idx += 1

        if self.current_hand_idx >= len(self.player_hands):
            return self._play_dealer_and_calculate_rewards()
        else:
            self.can_double = True
            self.can_surrender = True
            self.can_insure = True
            return self._get_state(), 0, False

    def _play_dealer_and_calculate_rewards(self):
        dealer_sum = self._get_hand_sum(self.dealer_hand)

        while dealer_sum < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_sum = self._get_hand_sum(self.dealer_hand)

        total_reward = 0
        dealer_busted = dealer_sum > 21
        hand_rewards = []

        for i, hand in enumerate(self.player_hands):
            if self.surrendered_hands[i]:
                hand_reward = -0.5
                hand_rewards.append(hand_reward)
                total_reward += hand_reward
                continue

            player_sum = self._get_hand_sum(hand)
            has_busted = player_sum > 21

            # Use enhanced reward function
            hand_reward = self.get_enhanced_reward(
                hand,
                self.dealer_hand,
                has_busted,
                hand_index=i,
            )

            hand_rewards.append(hand_reward)
            total_reward += hand_reward

        # Clear double down info for next game
        if hasattr(self, "double_down_info"):
            delattr(self, "double_down_info")

        self.game_over = True
        return self._get_state(), total_reward, True

    def get_game_info(self):
        info = {
            "deck_info": self.get_card_counting_info(),
            "player_hands": self.player_hands,
            "dealer_hand": self.dealer_hand,
            "current_hand_idx": self.current_hand_idx,
            "game_over": self.game_over,
            "games_played": self.games_played,
        }
        return info

    def get_detailed_win_stats(self):
        if not self.game_over:
            return None

        stats = {
            "total_hands": len(self.player_hands),
            "hands_won": 0,
            "hands_lost": 0,
            "hands_pushed": 0,
            "double_downs": sum(self.doubled_down),
            "splits": len(self.player_hands) - 1,
            "surrenders": sum(self.surrendered_hands),
            "insurance_bets": sum(1 for bet in self.insurance_bets if bet > 0),
            "blackjacks": 0,
            "busts": 0,
            "hand_details": [],
        }

        dealer_sum = self._get_hand_sum(self.dealer_hand)
        dealer_busted = dealer_sum > 21
        dealer_blackjack = len(self.dealer_hand) == 2 and dealer_sum == 21

        for i, hand in enumerate(self.player_hands):
            player_sum = self._get_hand_sum(hand)
            is_doubled = self.doubled_down[i]
            is_surrendered = self.surrendered_hands[i]
            insurance_bet = self.insurance_bets[i]

            hand_detail = {
                "hand_index": i,
                "cards": hand.copy(),
                "sum": player_sum,
                "doubled": is_doubled,
                "surrendered": is_surrendered,
                "insurance_bet": insurance_bet,
                "result": None,
                "reward": 0,
            }

            if is_surrendered:
                hand_detail["result"] = "surrender"
                hand_detail["reward"] = -0.5
                stats["hands_lost"] += 1
                stats["hand_details"].append(hand_detail)
                continue

            if player_sum > 21:
                hand_detail["result"] = "bust"
                hand_detail["reward"] = -1
                stats["busts"] += 1
                stats["hands_lost"] += 1
            elif len(hand) == 2 and player_sum == 21 and not dealer_blackjack:
                hand_detail["result"] = "blackjack"
                hand_detail["reward"] = 1.5
                stats["blackjacks"] += 1
                stats["hands_won"] += 1
            elif dealer_busted or player_sum > dealer_sum:
                hand_detail["result"] = "win"
                hand_detail["reward"] = 1
                stats["hands_won"] += 1
            elif player_sum < dealer_sum:
                hand_detail["result"] = "loss"
                hand_detail["reward"] = -1
                stats["hands_lost"] += 1
            else:
                hand_detail["result"] = "push"
                hand_detail["reward"] = 0
                stats["hands_pushed"] += 1

            if is_doubled:
                hand_detail["reward"] *= 2

            if insurance_bet > 0:
                if self._dealer_has_blackjack():
                    hand_detail["reward"] += insurance_bet * 2
                else:
                    hand_detail["reward"] -= insurance_bet

            stats["hand_details"].append(hand_detail)

        stats["win_rate"] = (
            stats["hands_won"] / stats["total_hands"] if stats["total_hands"] > 0 else 0
        )

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

        return stats
