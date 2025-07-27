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

    def _is_valid_action(self, action):
        if self.game_over or self.current_hand_idx >= len(self.player_hands):
            return False

        current_hand = self.player_hands[self.current_hand_idx]

        if action == 0:
            return True
        elif action == 1:
            return self._get_hand_sum(current_hand) < 21
        elif action == 2:
            return len(current_hand) == 2 and self.can_double
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

    def get_reward(self, agent_hand, dealer_hand, has_busted):
        agent_total = self._get_hand_sum(agent_hand)
        dealer_total = self._get_hand_sum(dealer_hand)

        if has_busted:
            return -3.0

        if agent_total == 21 and len(agent_hand) == 2:
            return 1.0

        if not self.game_over:
            if agent_total < 12:
                return 0.01
            elif 12 <= agent_total <= 16:
                if dealer_total >= 7:
                    return -0.01
                else:
                    return 0.01
            elif 17 <= agent_total <= 20:
                return 0.02
            else:
                return 0.0

        if dealer_total > 21:
            return 1.0

        if agent_total > dealer_total:
            return 1.0
        elif agent_total < dealer_total:
            return -1.0
        else:
            return 0.0

    def _get_state(self):
        if self.game_over or self.current_hand_idx >= len(self.player_hands):
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
        has_usable_ace = 11 in current_hand and self._get_hand_sum(current_hand) <= 21

        can_split = self._can_split() and len(self.player_hands) < 4
        can_double = self.can_double and len(current_hand) == 2
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
            if not (len(current_hand) == 2 and self.can_double):
                return self._get_state(), -0.1, False

            current_hand.append(self._draw_card())
            self.doubled_down[self.current_hand_idx] = True

            player_sum = self._get_hand_sum(current_hand)
            dealer_up = self.dealer_hand[0]

            shaping_reward = 0.0

            original_sum = self._get_hand_sum(current_hand[:-1])

            if original_sum == 11:
                shaping_reward = 0.1
            elif original_sum == 10 and dealer_up <= 9:
                shaping_reward = 0.1
            elif original_sum == 9 and dealer_up in [3, 4, 5, 6]:
                shaping_reward = 0.1
            elif original_sum == 8 and dealer_up in [5, 6]:
                shaping_reward = 0.05
            else:
                shaping_reward = -0.05

            next_state, final_reward, done = self._move_to_next_hand()
            return next_state, shaping_reward, done

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

            if self._dealer_has_blackjack():
                return self._get_state(), 1.0, False
            else:
                return self._get_state(), -0.5, False

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

        for i, hand in enumerate(self.player_hands):
            if self.surrendered_hands[i]:
                total_reward -= 0.5
                continue

            player_sum = self._get_hand_sum(hand)
            has_busted = player_sum > 21

            hand_reward = self.get_reward(hand, self.dealer_hand, has_busted)

            if self.doubled_down[i]:
                hand_reward *= 2

            if self.insurance_bets[i] > 0:
                if self._dealer_has_blackjack():
                    hand_reward += self.insurance_bets[i] * 2
                else:
                    hand_reward -= self.insurance_bets[i]

            total_reward += hand_reward

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
