import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models import Cage
from typing import List, Tuple
import random

class KillerSudokuEnv:
    """Environment for Killer Sudoku puzzle solving with RL"""
    
    def __init__(self, cages: List[Cage], solution: List[List[int]] = None):
        """
        cages: List of Cage objects from your generator
        solution: Optional answer grid for verification (9x9 list)
        """
        self.cages = cages
        self.solution = np.array(solution) if solution is not None else None
        self.reset()
        
    def reset(self):
        """Reset to initial empty state"""
        self.grid = np.zeros((9, 9), dtype=np.int32)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """Convert grid and cages to state representation"""
        # Channel 0: Current grid (normalized)
        grid_normalized = self.grid.astype(np.float32) / 9.0
        
        # Channel 1: Cage target sums (normalized)
        cage_sums = np.zeros((9, 9), dtype=np.float32)
        for cage in self.cages:
            for r, c in cage.cells:
                cage_sums[r, c] = cage.total / 45.0  # Normalize (max sum is 45)
        
        # Channel 2: Cage size encoding
        cage_sizes = np.zeros((9, 9), dtype=np.float32)
        for cage in self.cages:
            for r, c in cage.cells:
                cage_sizes[r, c] = len(cage) / 9.0  # Normalize
        
        # Stack all channels
        state = np.stack([grid_normalized, cage_sums, cage_sizes], axis=0)
        
        return state
    
    def _is_valid_placement(self, row: int, col: int, num: int, allow_overwrite: bool = True) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Store current value to temporarily remove it for checking
        current_value = self.grid[row, col]
        
        # If not allowing overwrites and cell is filled, reject
        if not allow_overwrite and current_value != 0:
            return False
        
        # If trying to place the same number that's already there, it's pointless
        if current_value == num:
            return False
        
        # Temporarily remove current value for constraint checking
        self.grid[row, col] = 0
        
        # Check row constraint
        if num in self.grid[row]:
            self.grid[row, col] = current_value  # Restore
            return False
        
        # Check column constraint
        if num in self.grid[:, col]:
            self.grid[row, col] = current_value  # Restore
            return False
        
        # Check 3x3 box constraint
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if num in self.grid[box_r:box_r+3, box_c:box_c+3]:
            self.grid[row, col] = current_value  # Restore
            return False
        
        # Restore original value
        self.grid[row, col] = current_value
        return True
    
    def _find_cage_for_cell(self, row: int, col: int) -> Cage:
        """Find which cage a cell belongs to"""
        for cage in self.cages:
            if (row, col) in cage.cells:
                return cage
        return None
    
    def _check_cage_constraint(self, cage: Cage) -> bool:
        """Check if a cage's sum constraint is satisfied or still possible"""
        values = [self.grid[r, c] for r, c in cage.cells]
        filled = [v for v in values if v != 0]
        empty_count = len(values) - len(filled)
        
        current_sum = sum(filled)
        
        # If all filled, must equal target
        if empty_count == 0:
            return current_sum == cage.total
        
        # Check if sum already exceeded
        if current_sum >= cage.total:
            return False
        
        # Check if still possible to reach target
        remaining = cage.total - current_sum
        min_possible = sum(range(1, empty_count + 1))
        max_possible = sum(range(9, 9 - empty_count, -1))
        
        return min_possible <= remaining <= max_possible
    
    def step(self, action: Tuple[int, int, int]):
        """
        action: (row, col, value) tuple
        Returns: (next_state, reward, done, info)
        """
        row, col, value = action
        self.steps += 1
        
        # Store old value for potential revert
        old_value = self.grid[row, col]
        
        # Check if action is valid (allows overwrites)
        if not self._is_valid_placement(row, col, value, allow_overwrite=True):
            return self._get_state(), -10.0, False, {'invalid': True}
        
        # Place the number
        self.grid[row, col] = value
        
        # Find the cage this cell belongs to
        cell_cage = self._find_cage_for_cell(row, col)
        
        # Check cage constraint
        if cell_cage and not self._check_cage_constraint(cell_cage):
            # Constraint violated - revert and penalize
            self.grid[row, col] = old_value
            return self._get_state(), -5.0, False, {'constraint_violation': True}
        
        # Calculate reward
        reward = 0.0
        completed_cages = 0
        
        # Give small bonus for overwriting (correcting a mistake)
        if old_value != 0 and self.solution[row, col] == value:
            reward += 3.0
        else:
            reward -= 5.0
        
        # Check if cage is completed correctly
        if cell_cage:
            cage_values = [self.grid[r, c] for r, c in cell_cage.cells]
            if all(v != 0 for v in cage_values):
                cage_sum = sum(cage_values)
                if cage_sum == cell_cage.total:
                    completed_cages += 1
                    reward += 2.0  # Bonus for completing a cage
        
        # Check if puzzle is solved
        done = np.all(self.grid != 0)
        
        if done:
            # Verify solution is correct
            if self._verify_solution():
                reward = 100.0
            else:
                reward = -50.0  # Major penalty for invalid solution
        else:
            # Small positive reward for valid placement
            reward += 0.1
            # Reward for completing cages
            if completed_cages > 0:
                reward += completed_cages
            # Small penalty for each step to encourage efficiency
            reward -= 0.01
        
        return self._get_state(), reward, done, {'completed_cages': completed_cages}
    
    def _verify_solution(self) -> bool:
        """Verify if the current grid is a valid solution"""
        # Check all rows have 1-9
        for row in self.grid:
            if sorted(row) != list(range(1, 10)):
                return False
        
        # Check all columns have 1-9
        for col in range(9):
            if sorted(self.grid[:, col]) != list(range(1, 10)):
                return False
        
        # Check all 3x3 boxes have 1-9
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                box = self.grid[box_r:box_r+3, box_c:box_c+3].flatten()
                if sorted(box) != list(range(1, 10)):
                    return False
        
        # Check all cages sum correctly
        for cage in self.cages:
            cage_sum = sum(self.grid[r, c] for r, c in cage.cells)
            if cage_sum != cage.total:
                return False
        
        return True
    
    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        """Return list of valid actions (allows overwrites)"""
        valid_actions = []
        for row in range(9):
            for col in range(9):
                for num in range(1, 10):
                    if self._is_valid_placement(row, col, num, allow_overwrite=True):
                        valid_actions.append((row, col, num))
        return valid_actions
    
    def get_smart_valid_actions(self) -> List[Tuple[int, int, int]]:
        """Return valid actions with cage constraint pre-filtering (allows overwrites)"""
        valid_actions = []
        for row in range(9):
            for col in range(9):
                cage = self._find_cage_for_cell(row, col)
                old_value = self.grid[row, col]
                
                for num in range(1, 10):
                    if not self._is_valid_placement(row, col, num, allow_overwrite=True):
                        continue
                    
                    # Check if this number would keep cage constraint valid
                    if cage:
                        # Temporarily place number
                        self.grid[row, col] = num
                        is_valid = self._check_cage_constraint(cage)
                        self.grid[row, col] = old_value  # Revert
                        
                        if is_valid:
                            valid_actions.append((row, col, num))
                    else:
                        valid_actions.append((row, col, num))
        
        return valid_actions


class SudokuDQN(nn.Module):
    """Deep Q-Network for Killer Sudoku"""
    
    def __init__(self, action_size=729):
        super(SudokuDQN, self).__init__()
        
        # Convolutional layers to extract spatial features (now 3 input channels)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 9 * 9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """DQN Agent for training"""
    
    def __init__(self, state_shape=(3, 9, 9), action_size=729, lr=0.0001):
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Q-Networks
        self.policy_net = SudokuDQN(action_size).to(self.device)
        self.target_net = SudokuDQN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """Choose action using epsilon-greedy policy"""
        if not valid_actions:
            return None
        
        if np.random.rand() <= self.epsilon:
            # Random valid action
            return random.choice(valid_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        action_mask = np.full(self.action_size, -np.inf)
        for r, c, v in valid_actions:
            action_idx = r * 81 + c * 9 + (v - 1)
            action_mask[action_idx] = q_values[action_idx]
        
        # Choose best valid action
        best_action_idx = np.argmax(action_mask)
        r = best_action_idx // 81
        c = (best_action_idx % 81) // 9
        v = (best_action_idx % 9) + 1
        
        return (r, c, v)
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([s for s, a, r, ns, d in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s, a, r, ns, d in batch])).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
        
        targets = current_q_values.clone()
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            r, c, v = action
            action_idx = r * 81 + c * 9 + (v - 1)
            
            if done:
                targets[i, action_idx] = reward
            else:
                targets[i, action_idx] = reward + self.gamma * next_q_values[i].max()
        
        # Compute loss and update
        loss = self.criterion(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_agent(generator, agent, episodes=1000, max_steps=200, use_smart_actions=True):
    """
    Train the DQN agent
    
    generator: Your SudokuGenerator instance
    agent: DQNAgent instance
    episodes: Number of training episodes
    max_steps: Maximum steps per episode
    use_smart_actions: Use cage-aware action filtering
    """
    scores = []
    losses = []
    solved_count = 0
    
    for episode in range(episodes):
        # Generate new puzzle
        cages, solution = generator.puzzle()
        env = KillerSudokuEnv(cages, solution)
        
        state = env.reset()
        total_reward = 0
        episode_losses = []
        done = False
        
        for _ in range(max_steps):
            # Get valid actions
            if use_smart_actions:
                valid_actions = env.get_smart_valid_actions()
            else:
                valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose and take action
            action = agent.act(state, valid_actions)
            if action is None:
                break
            
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            if done:
                if reward > 50:  # Successfully solved
                    solved_count += 1
                break
        
        scores.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Logging
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses) if losses else 0
            solve_rate = solved_count / (episode + 1) * 100
            
            print(f"Episode {episode}/{episodes}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Solve Rate: {solve_rate:.2f}%")
            print(f"  Total Solved: {solved_count}")
            print()
    
    return agent, scores, losses


def train_with_visualization(generator, agent, game, ui, episodes=1000, max_steps=200, 
                            use_smart_actions=True, delay_ms=100):
    """
    Train with live visualization in the UI
    
    generator: Your SudokuGenerator instance
    agent: DQNAgent instance
    game: Game instance
    ui: KillerSudokuApp instance
    episodes: Number of training episodes
    max_steps: Maximum steps per episode
    use_smart_actions: Use cage-aware action filtering
    delay_ms: Delay between moves in milliseconds
    """
    scores = []
    losses = []
    solved_count = [0]  # Use list to allow modification in nested function
    current_episode = [0]  # Use list to allow modification in nested function
    
    def train_episode():
        episode = current_episode[0]
        
        if episode >= episodes:
            print("\nTraining complete!")
            print(f"Final solve rate: {solved_count[0] / episodes * 100:.2f}%")
            return
        
        # Generate new puzzle
        cages, solution = generator.puzzle()
        
        # Update game with new puzzle
        game.reset()
        game.cages = cages
        game.ans = solution
        game.grid = [[0 for _ in range(9)] for _ in range(9)]
        game.selected_cell = None
        ui.cage_data = game.cage_data()
        ui.redraw()
        
        # Create environment
        env = KillerSudokuEnv(cages, solution)
        state = env.reset()
        
        episode_data = {
            'total_reward': 0,
            'episode_losses': [],
            'step': 0,
            'done': False,
            'state': state
        }
        
        def train_step():
            if episode_data['done'] or episode_data['step'] >= max_steps:
                # Episode finished
                scores.append(episode_data['total_reward'])
                if episode_data['episode_losses']:
                    losses.append(np.mean(episode_data['episode_losses']))
                
                # Update target network periodically
                if episode % 10 == 0:
                    agent.update_target_network()
                
                # Logging
                if episode % 10 == 0:
                    avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                    avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses) if losses else 0
                    solve_rate = solved_count[0] / (episode + 1) * 100
                    
                    print(f"Episode {episode}/{episodes} - Score: {episode_data['total_reward']:.2f}, "
                          f"Epsilon: {agent.epsilon:.3f}, Solve Rate: {solve_rate:.2f}%")
                
                # Move to next episode
                current_episode[0] += 1
                ui.root.after(1000, train_episode)  # Wait 1 second before next episode
                return
            
            # Get valid actions
            if use_smart_actions:
                valid_actions = env.get_smart_valid_actions()
            else:
                valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                episode_data['done'] = True
                ui.root.after(10, train_step)
                return
            
            # Choose action
            action = agent.act(episode_data['state'], valid_actions)
            if action is None:
                episode_data['done'] = True
                ui.root.after(10, train_step)
                return
            
            row, col, value = action
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update UI
            game.select(row, col)
            game.enter_number(value)
            ui.redraw()
            
            # Store experience
            agent.remember(episode_data['state'], action, reward, next_state, done)
            
            # Update episode data
            episode_data['total_reward'] += reward
            episode_data['step'] += 1
            episode_data['state'] = next_state
            
            # Train
            loss = agent.replay()
            if loss > 0:
                episode_data['episode_losses'].append(loss)
            
            if done:
                if reward > 50:  # Successfully solved
                    solved_count[0] += 1
                episode_data['done'] = True
            
            # Schedule next step
            ui.root.after(delay_ms, train_step)
        
        # Start first step
        train_step()
    
    # Start training
    train_episode()

    agent.save("killer_sudoku_model.pth")


def solve_with_visualization(agent, game, ui, max_steps=200, delay_ms=500):
    """
    Use trained agent to solve current puzzle with visualization
    
    agent: Trained DQNAgent instance
    game: Game instance with loaded puzzle
    ui: KillerSudokuApp instance
    max_steps: Maximum steps to attempt
    delay_ms: Delay between moves in milliseconds
    """
    # Create environment from current game state
    env = KillerSudokuEnv(game.cages, game.ans)
    
    # Copy current grid state to environment
    for r in range(9):
        for c in range(9):
            if game.grid[r][c] != 0:
                env.grid[r][c] = game.grid[r][c]
    
    state = env._get_state()
    step_count = [0]
    
    # Disable exploration for solving
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    def solve_step():
        if step_count[0] >= max_steps:
            print("Max steps reached!")
            agent.epsilon = old_epsilon
            return
        
        # Get valid actions
        valid_actions = env.get_smart_valid_actions()
        
        if not valid_actions:
            if np.all(env.grid != 0):
                print("Puzzle solved!")
            else:
                print("No valid actions available!")
            agent.epsilon = old_epsilon
            return
        
        # Choose best action (no exploration)
        action = agent.act(state, valid_actions)
        if action is None:
            print("Agent returned no action!")
            agent.epsilon = old_epsilon
            return
        
        row, col, value = action
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Update UI
        game.select(row, col)
        game.enter_number(value)
        ui.redraw()
        
        step_count[0] += 1
        
        if done:
            if reward > 50:
                print(f"Successfully solved in {step_count[0]} steps!")
            else:
                print(f"Failed to solve correctly!")
            agent.epsilon = old_epsilon
            return
        
        # Copy state for next iteration
        state_copy = next_state.copy()
        
        # Schedule next step
        ui.root.after(delay_ms, lambda: solve_step_with_state(state_copy))
    
    def solve_step_with_state(current_state):
        nonlocal state
        state = current_state
        solve_step()
    
    # Start solving
    print("Starting AI solver...")
    solve_step()

"""
# Example usage:
if __name__ == "__main__":
    from generator import SudokuGenerator
    
    # Initialize
    generator = SudokuGenerator()
    agent = DQNAgent(state_shape=(3, 9, 9), lr=0.0001)
    
    # Option 1: Train without visualization (faster)
    print("Starting training without visualization...")
    trained_agent, scores, losses = train_agent(
        generator, 
        agent, 
        episodes=2000,
        max_steps=200,
        use_smart_actions=True
    )
    
    # Save model
    trained_agent.save("killer_sudoku_model.pth")
    print("Model saved!")
"""

# Example usage WITH VISUALIZATION:

import tkinter as tk
from game import Game
from ui_tk import KillerSudokuApp
from generator import SudokuGenerator

def main_with_training_visualization():
    # Setup UI
    g = SudokuGenerator()
    cages, ans = g.puzzle()
    root = tk.Tk()
    game = Game(starters={}, cages=cages, ans=ans)
    ui = KillerSudokuApp(root, game)
    
    # Create agent
    agent = DQNAgent(state_shape=(3, 9, 9), lr=0.0001)
    
    # Train with visualization (will run in UI loop)
    train_with_visualization(
        g, 
        agent, 
        game, 
        ui, 
        episodes=1000,  # Fewer episodes since it's slow with visualization
        max_steps=500,
        use_smart_actions=True,
        delay_ms=50  # 50ms between moves
    )
    
    root.mainloop()


def main_with_solver_visualization():
    # Setup UI
    g = SudokuGenerator()
    cages, ans = g.puzzle()
    root = tk.Tk()
    game = Game(starters={}, cages=cages, ans=ans)
    ui = KillerSudokuApp(root, game)
    
    # Load trained agent
    agent = DQNAgent(state_shape=(3, 9, 9))
    agent.load("killer_sudoku_model.pth")
    
    # Add button to trigger solving
    solve_btn = tk.Button(
        root, 
        text="Solve with AI", 
        command=lambda: solve_with_visualization(agent, game, ui, delay_ms=500)
    )
    solve_btn.pack()
    
    root.mainloop()

if __name__ == "__main__":
    # Choose one:
    main_with_training_visualization()  # Watch agent learn
    # main_with_solver_visualization()  # Watch agent solve
