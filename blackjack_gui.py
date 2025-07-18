#!/usr/bin/env python3
"""
Blackjack RL Agent GUI - Clean and Simple Version

A straightforward GUI for visualizing trained reinforcement learning agents playing Blackjack.
"""

import pygame
import sys
import os
import random
import time

AGENT_PATH = "curriculum_agent_dqn_0.pth"

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

try:
    from scripts.BlackJackENV import BlackjackEnv
    from scripts.RLAgent import DQNAgent, QLearningAgent
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Initialize Pygame
pygame.init()

# Colors
COLORS = {
    "background": (34, 139, 34),
    "card_white": (255, 255, 255),
    "card_black": (0, 0, 0),
    "card_red": (220, 20, 60),
    "card_back": (30, 144, 255),
    "button": (70, 130, 180),
    "button_hover": (100, 149, 237),
    "button_disabled": (128, 128, 128),
    "text_white": (255, 255, 255),
    "text_gold": (255, 215, 0),
    "panel": (60, 60, 60),
    "border": (200, 200, 200),
}

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CARD_WIDTH = 70
CARD_HEIGHT = 100


class SimpleCard:
    """Simple card representation."""

    def __init__(self, value):
        self.value = value
        self.suit = random.choice(["♠", "♥", "♦", "♣"])

    def get_display_value(self):
        if self.value == 1:
            return "A"
        elif self.value == 11:
            return "J"
        elif self.value == 12:
            return "Q"
        elif self.value == 13:
            return "K"
        else:
            return str(self.value)

    def get_text_color(self):
        if self.suit in ["♥", "♦"]:
            return COLORS["card_red"]
        return COLORS["card_black"]


class SimpleButton:
    """Simple button class."""

    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.enabled = True
        self.hovered = False

    def draw(self, screen):
        color = (
            COLORS["button_disabled"]
            if not self.enabled
            else (COLORS["button_hover"] if self.hovered else COLORS["button"])
        )
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, COLORS["border"], self.rect, 2)

        text_surface = self.font.render(self.text, True, COLORS["text_white"])
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.enabled and self.rect.collidepoint(event.pos):
                return True
        return False


class BlackjackGUI:
    """Simple Blackjack GUI."""

    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Blackjack RL Agent - Simple GUI")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game components
        self.env = None
        self.agent = None

        # Game state
        self.state = "menu"  # menu, playing, game_over
        self.current_game = 0
        self.auto_play = False
        self.last_auto_action = 0

        # Game data
        self.player_hands = []  # Support multiple hands for splits
        self.dealer_cards = []
        self.player_sums = []  # Sum for each hand
        self.dealer_sum = 0
        self.dealer_hidden = True
        self.game_result = ""
        self.current_hand_idx = 0  # Which hand is currently being played
        self.bet_amounts = []  # Bet amount for each hand
        self.doubled_hands = []  # Track which hands were doubled
        self.split_from_pair = None  # Track what pair was split

        # Statistics
        self.stats = {"games": 0, "wins": 0, "losses": 0, "pushes": 0}

        # Setup UI
        self.setup_buttons()

    def setup_buttons(self):
        """Setup all buttons."""
        # Menu buttons
        self.menu_buttons = {
            "load_agent": SimpleButton(
                500, 300, 200, 50, "Load Agent", self.font_medium
            ),
            "quit": SimpleButton(500, 370, 200, 50, "Quit", self.font_medium),
        }

        # Game buttons
        self.game_buttons = {
            "hit": SimpleButton(200, 700, 100, 50, "Hit", self.font_medium),
            "stand": SimpleButton(320, 700, 100, 50, "Stand", self.font_medium),
            "double": SimpleButton(440, 700, 100, 50, "Double", self.font_medium),
            "split": SimpleButton(560, 700, 100, 50, "Split", self.font_medium),
            "new_game": SimpleButton(680, 700, 120, 50, "New Game", self.font_medium),
            "auto_play": SimpleButton(820, 700, 120, 50, "Auto Play", self.font_medium),
            "menu": SimpleButton(960, 700, 100, 50, "Menu", self.font_medium),
        }

    def load_agent(self):
        """Load the trained agent."""
        try:
            print("Loading agent...")

            # Try to load curriculum agent first
            agent_path = AGENT_PATH
            if os.path.exists(agent_path):
                print(f"Loading {agent_path}")
                self.agent = DQNAgent(action_space=[0, 1, 2, 3])
                self.agent.load_model(agent_path)
                self.agent.epsilon = 0.0
            else:
                print("Creating demo agent...")
                self.agent = DQNAgent(action_space=[0, 1, 2, 3])
                # Quick training
                env = BlackjackEnv(deck_type="infinite", budget=100)
                for i in range(500):
                    state = env.reset()
                    done = False
                    while not done:
                        action = self.agent.get_action(state)
                        next_state, reward, done = env.step(action)
                        self.agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        if len(self.agent.memory) > 32:
                            self.agent.replay()
                    if i % 100 == 0:
                        print(f"Training: {i}/500")

            self.env = BlackjackEnv(deck_type="infinite", budget=100)
            print("Agent loaded successfully!")
            self.state = "playing"
            self.new_game()

        except Exception as e:
            print(f"Error loading agent: {e}")

    def new_game(self):
        """Start a new game."""
        if not self.env or not self.agent:
            print("No agent loaded!")
            return

        try:
            # Reset environment
            self.env.reset()
            self.current_game += 1

            # Get initial game state
            game_info = self.env.get_game_info()
            if game_info:
                self.player_hands = (
                    game_info["player_hands"] if game_info["player_hands"] else []
                )
                self.dealer_cards = game_info["dealer_hand"]
                self.player_sums = [
                    self.calculate_hand_value(hand) for hand in self.player_hands
                ]
                self.dealer_sum = self.calculate_hand_value(self.dealer_cards)
                self.dealer_hidden = True
                self.game_result = ""
                self.current_hand_idx = 0
                self.bet_amounts = [10] * len(self.player_hands)  # Default bet
                self.doubled_hands = []
                self.split_from_pair = None

                print(f"New game {self.current_game} started!")
                print(
                    f"Player: {self.player_sums[self.current_hand_idx]}, Dealer: {self.dealer_cards[1] if len(self.dealer_cards) > 1 else '?'}"
                )

                self.state = "playing"

        except Exception as e:
            print(f"Error starting new game: {e}")

    def calculate_hand_value(self, cards):
        """Calculate hand value."""
        total = 0
        aces = 0

        for card in cards:
            if card == 1:
                aces += 1
                total += 11
            elif card > 10:
                total += 10
            else:
                total += card

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    def perform_action(self, action):
        """Perform a game action."""
        if not self.env or not self.agent or self.state != "playing":
            return

        try:
            action_names = ["Stand", "Hit", "Double", "Split"]
            print(f"Action: {action_names[action]}")

            # Handle special actions with animations
            if action == 2:  # Double
                self.handle_double_action()
            elif action == 3:  # Split
                self.handle_split_action()

            # Perform action in environment
            state, reward, done = self.env.step(action)

            # Update game state
            game_info = self.env.get_game_info()
            if game_info:
                self.player_hands = (
                    game_info["player_hands"] if game_info["player_hands"] else []
                )
                self.dealer_cards = game_info["dealer_hand"]
                self.player_sums = [
                    self.calculate_hand_value(hand) for hand in self.player_hands
                ]

                print(f"Player sum: {self.player_sums[self.current_hand_idx]}")

                # Check if current hand is done
                if self.player_sums[self.current_hand_idx] > 21:
                    print(f"Hand {self.current_hand_idx + 1} busted!")
                    self.move_to_next_hand()
                elif done:
                    self.end_game(reward)

        except Exception as e:
            print(f"Error performing action: {e}")

    def handle_double_action(self):
        """Handle double down action with visual feedback."""
        print(f"🎯 DOUBLE DOWN on hand {self.current_hand_idx + 1}!")
        print(
            f"💰 Bet doubled from ${self.bet_amounts[self.current_hand_idx]} to ${self.bet_amounts[self.current_hand_idx] * 2}"
        )

        # Double the bet
        self.bet_amounts[self.current_hand_idx] *= 2
        self.doubled_hands.append(self.current_hand_idx)

        # Visual feedback
        self.draw_game()
        pygame.display.flip()
        pygame.time.wait(1000)  # Show double down message

    def handle_split_action(self):
        """Handle split action with visual feedback."""
        if self.player_hands and len(self.player_hands[self.current_hand_idx]) >= 2:
            card_value = self.player_hands[self.current_hand_idx][0]
            self.split_from_pair = card_value

            print(f"✂️ SPLIT PAIR of {card_value}s!")
            print(
                f"💰 Additional bet of ${self.bet_amounts[self.current_hand_idx]} placed"
            )

            # Add bet for new hand
            self.bet_amounts.append(self.bet_amounts[self.current_hand_idx])

            # Visual feedback
            self.draw_game()
            pygame.display.flip()
            pygame.time.wait(1000)  # Show split message

    def move_to_next_hand(self):
        """Move to the next hand if playing multiple hands."""
        if self.current_hand_idx < len(self.player_hands) - 1:
            self.current_hand_idx += 1
            print(f"🎮 Moving to hand {self.current_hand_idx + 1}")

            # Visual feedback for hand change
            self.draw_game()
            pygame.display.flip()
            pygame.time.wait(1000)
        else:
            print("🎯 All hands completed!")
            self.end_game(0)  # End game when all hands are done

    def end_game(self, reward):
        """End the current game."""
        print("Game ended!")

        # Reveal dealer cards
        self.dealer_hidden = False
        self.dealer_sum = self.calculate_hand_value(self.dealer_cards)

        # Simulate dealer drawing (simple version)
        while self.dealer_sum < 17:
            new_card = random.randint(1, 11)
            self.dealer_cards.append(new_card)
            self.dealer_sum = self.calculate_hand_value(self.dealer_cards)
            print(f"Dealer draws: {new_card}, total: {self.dealer_sum}")

        # Calculate results for all hands
        total_winnings = 0
        hand_results = []

        for i, hand in enumerate(self.player_hands):
            hand_sum = self.calculate_hand_value(hand)
            bet = self.bet_amounts[i] if i < len(self.bet_amounts) else 10

            # Determine result for this hand
            if hand_sum > 21:
                result = "BUST"
                winnings = -bet
            elif self.dealer_sum > 21:
                result = "WIN (Dealer Bust)"
                winnings = bet
            elif hand_sum > self.dealer_sum:
                result = "WIN"
                winnings = bet
            elif self.dealer_sum > hand_sum:
                result = "LOSE"
                winnings = -bet
            else:
                result = "PUSH"
                winnings = 0

            # Check for blackjack bonus
            if len(hand) == 2 and hand_sum == 21:
                result += " (Blackjack!)"
                if winnings > 0:
                    winnings = int(winnings * 1.5)  # 3:2 payout

            total_winnings += winnings
            hand_results.append((i + 1, hand_sum, result, winnings))

            print(f"Hand {i + 1}: {hand_sum} - {result} (${winnings:+})")

        # Set overall game result
        if total_winnings > 0:
            self.game_result = f"TOTAL WIN: ${total_winnings:+}"
        elif total_winnings < 0:
            self.game_result = f"TOTAL LOSS: ${total_winnings:+}"
        else:
            self.game_result = "PUSH (Break Even)"

        # Update statistics based on total result
        self.stats["games"] += 1
        if total_winnings > 0:
            self.stats["wins"] += 1
        elif total_winnings < 0:
            self.stats["losses"] += 1
        else:
            self.stats["pushes"] += 1

        print(f"Final Result: {self.game_result}")
        print(f"Dealer: {self.dealer_sum}")

        # Store hand results for display
        self.hand_results = hand_results

        self.state = "game_over"

    def draw_card(self, card_value, x, y, hidden=False):
        """Draw a single card."""
        if hidden:
            # Draw card back
            pygame.draw.rect(
                self.screen, COLORS["card_back"], (x, y, CARD_WIDTH, CARD_HEIGHT)
            )
            pygame.draw.rect(
                self.screen, COLORS["card_black"], (x, y, CARD_WIDTH, CARD_HEIGHT), 2
            )
        else:
            # Draw card face
            card = SimpleCard(card_value)
            pygame.draw.rect(
                self.screen, COLORS["card_white"], (x, y, CARD_WIDTH, CARD_HEIGHT)
            )
            pygame.draw.rect(
                self.screen, COLORS["card_black"], (x, y, CARD_WIDTH, CARD_HEIGHT), 2
            )

            # Draw value
            text = self.font_medium.render(
                card.get_display_value(), True, card.get_text_color()
            )
            self.screen.blit(text, (x + 10, y + 10))

            # Draw suit
            suit_text = self.font_small.render(card.suit, True, card.get_text_color())
            self.screen.blit(suit_text, (x + 10, y + 40))

    def draw_hand(
        self,
        cards,
        x,
        y,
        title,
        hidden_first=False,
        is_current=False,
        bet_amount=None,
        is_doubled=False,
    ):
        """Draw a hand of cards with betting information."""
        # Draw title with current hand indicator
        title_color = COLORS["text_gold"] if is_current else COLORS["text_white"]
        title_text = f"{title} {'← CURRENT' if is_current else ''}"
        title_surface = self.font_medium.render(title_text, True, title_color)
        self.screen.blit(title_surface, (x, y - 60))

        # Draw betting information
        if bet_amount:
            bet_text = f"Bet: ${bet_amount}"
            if is_doubled:
                bet_text += " (DOUBLED)"
            bet_color = COLORS["text_gold"] if is_doubled else COLORS["text_white"]
            bet_surface = self.font_small.render(bet_text, True, bet_color)
            self.screen.blit(bet_surface, (x, y - 40))

        # Draw cards
        for i, card in enumerate(cards):
            card_x = x + i * (CARD_WIDTH + 10)
            hidden = hidden_first and i == 0
            self.draw_card(card, card_x, y, hidden)

        # Draw hand value
        if not (hidden_first and self.dealer_hidden):
            value = self.calculate_hand_value(cards)
            value_text = self.font_small.render(
                f"Value: {value}", True, COLORS["text_white"]
            )
            self.screen.blit(value_text, (x, y + CARD_HEIGHT + 10))

    def draw_game(self):
        """Draw the game screen."""
        self.screen.fill(COLORS["background"])

        # Title
        title = self.font_large.render("Blackjack RL Agent", True, COLORS["text_white"])
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

        if self.state in ["playing", "game_over"]:
            # Draw dealer hand
            self.draw_hand(
                self.dealer_cards, 100, 100, "Dealer", hidden_first=self.dealer_hidden
            )

            # Draw player hands (multiple hands for splits)
            if self.player_hands:
                for i, hand in enumerate(self.player_hands):
                    hand_x = 100 + i * 250  # Space hands horizontally
                    hand_title = (
                        f"Hand {i + 1}" if len(self.player_hands) > 1 else "Player"
                    )
                    is_current = i == self.current_hand_idx
                    bet_amount = (
                        self.bet_amounts[i] if i < len(self.bet_amounts) else None
                    )
                    is_doubled = i in self.doubled_hands

                    self.draw_hand(
                        hand,
                        hand_x,
                        300,
                        hand_title,
                        is_current=is_current,
                        bet_amount=bet_amount,
                        is_doubled=is_doubled,
                    )

            # Draw split information
            if self.split_from_pair:
                split_text = f"Split from pair of {self.split_from_pair}s"
                split_surface = self.font_small.render(
                    split_text, True, COLORS["text_gold"]
                )
                self.screen.blit(split_surface, (100, 480))

            # Draw game info
            game_info = self.env.get_game_info()
            budget_info = f"Budget: ${game_info['budget']:.1f}" if 'budget' in game_info else "Budget: $100.0"
            info_texts = [
                f"Game: {self.current_game}",
                (
                    f"Current Hand: {self.current_hand_idx + 1}/{len(self.player_hands)}"
                    if len(self.player_hands) > 1
                    else f"Player Sum: {self.player_sums[self.current_hand_idx] if self.player_sums else 0}"
                ),
                f"Dealer Sum: {self.dealer_sum if not self.dealer_hidden else '?'}",
                f"Auto Play: {'ON' if self.auto_play else 'OFF'}",
                budget_info,
            ]

            for i, text in enumerate(info_texts):
                info_text = self.font_small.render(text, True, COLORS["text_white"])
                self.screen.blit(info_text, (600, 100 + i * 30))

            # Draw statistics
            stats_texts = [
                f"Games: {self.stats['games']}",
                f"Wins: {self.stats['wins']}",
                f"Losses: {self.stats['losses']}",
                f"Pushes: {self.stats['pushes']}",
                f"Win Rate: {(self.stats['wins'] / max(1, self.stats['games']) * 100):.1f}%",
            ]

            for i, text in enumerate(stats_texts):
                stats_text = self.font_small.render(text, True, COLORS["text_gold"])
                self.screen.blit(stats_text, (800, 100 + i * 30))

            # Draw game result
            if self.game_result:
                result_text = self.font_medium.render(
                    self.game_result, True, COLORS["text_gold"]
                )
                self.screen.blit(result_text, (100, 520))

            # Draw detailed hand results if game is over
            if self.state == "game_over" and hasattr(self, "hand_results"):
                results_title = self.font_small.render(
                    "Hand Results:", True, COLORS["text_white"]
                )
                self.screen.blit(results_title, (100, 550))

                for i, (hand_num, hand_sum, result, winnings) in enumerate(
                    self.hand_results
                ):
                    result_color = (
                        COLORS["text_gold"] if winnings >= 0 else COLORS["card_red"]
                    )
                    result_line = f"Hand {hand_num}: {hand_sum} - {result}"
                    result_surface = self.font_small.render(
                        result_line, True, result_color
                    )
                    self.screen.blit(result_surface, (120, 570 + i * 20))

            # Draw action buttons
            self.draw_action_buttons()

        else:
            # Draw menu
            menu_text = self.font_large.render(
                "Welcome to Blackjack RL Agent", True, COLORS["text_white"]
            )
            self.screen.blit(
                menu_text, (SCREEN_WIDTH // 2 - menu_text.get_width() // 2, 200)
            )

            instructions = [
                "Click 'Load Agent' to start",
                "Agent will be loaded automatically",
                "Use buttons to play or enable auto-play",
                "Split and Double actions are fully supported",
            ]

            for i, instruction in enumerate(instructions):
                inst_text = self.font_medium.render(
                    instruction, True, COLORS["text_white"]
                )
                self.screen.blit(
                    inst_text,
                    (SCREEN_WIDTH // 2 - inst_text.get_width() // 2, 250 + i * 30),
                )

            # Draw menu buttons
            for button in self.menu_buttons.values():
                button.draw(self.screen)

    def draw_action_buttons(self):
        """Draw action buttons with proper state management."""
        for button in self.game_buttons.values():
            # Enable/disable buttons based on game state
            if self.state == "playing":
                if button.text in ["Hit", "Stand", "Double", "Split"]:
                    button.enabled = True

                    # Disable hit if current hand has 21 or more
                    if (
                        button.text == "Hit"
                        and self.player_sums
                        and self.player_sums[self.current_hand_idx] >= 21
                    ):
                        button.enabled = False

                    # Disable double if already doubled or not first action
                    if button.text == "Double":
                        # Check if this is the first action on this hand
                        current_hand = (
                            self.player_hands[self.current_hand_idx]
                            if self.player_hands
                            else []
                        )
                        button.enabled = (
                            len(current_hand) == 2
                        )  # Only allow double on first action

                    # Disable split if not a pair or already split
                    if button.text == "Split":
                        current_hand = (
                            self.player_hands[self.current_hand_idx]
                            if self.player_hands
                            else []
                        )
                        if len(current_hand) == 2:
                            button.enabled = (
                                current_hand[0] == current_hand[1]
                            )  # Only pairs
                        else:
                            button.enabled = False
                else:
                    button.enabled = True
            else:  # game_over
                button.enabled = button.text in ["New Game", "Auto Play", "Menu"]

            button.draw(self.screen)

    def handle_events(self):
        """Handle all events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.auto_play = not self.auto_play
                    print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")
                elif event.key == pygame.K_ESCAPE:
                    self.state = "menu"

            else:
                # Handle button clicks
                if self.state == "menu":
                    for name, button in self.menu_buttons.items():
                        if button.handle_event(event):
                            if name == "load_agent":
                                self.load_agent()
                            elif name == "quit":
                                return False

                elif self.state in ["playing", "game_over"]:
                    for name, button in self.game_buttons.items():
                        if button.handle_event(event):
                            if name == "hit":
                                self.perform_action(1)
                            elif name == "stand":
                                self.perform_action(0)
                            elif name == "double":
                                self.perform_action(2)
                            elif name == "split":
                                self.perform_action(3)
                            elif name == "new_game":
                                self.new_game()
                            elif name == "auto_play":
                                self.auto_play = not self.auto_play
                                print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")
                            elif name == "menu":
                                self.state = "menu"

        return True

    def update_auto_play(self):
        """Handle auto-play logic."""
        if not self.auto_play or self.state != "playing" or not self.agent:
            return

        current_time = time.time()
        if current_time - self.last_auto_action > 1.0:  # 1 second delay
            try:
                # Get agent action
                state = self.env._get_state()
                action = self.agent.get_action(state)

                # Don't hit if player has 21 or more
                if self.player_sums and self.player_sums[self.current_hand_idx] >= 21:
                    action = 0  # Force stand

                self.perform_action(action)
                self.last_auto_action = current_time

            except Exception as e:
                print(f"Auto-play error: {e}")

    def run(self):
        """Main game loop."""
        running = True

        while running:
            running = self.handle_events()
            self.update_auto_play()
            self.draw_game()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Main function."""
    print("Starting Simple Blackjack RL Agent GUI...")
    print("Controls:")
    print("  - SPACE: Toggle auto-play")
    print("  - ESC: Return to menu")
    print("  - Click buttons to play")

    gui = BlackjackGUI()
    gui.run()


if __name__ == "__main__":
    main()
