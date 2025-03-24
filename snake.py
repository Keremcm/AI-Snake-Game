import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Oyun Parametreleri
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20

# Renkler
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Hareket Yönleri
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DQN Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        return self.get_state()
    
    def spawn_food(self):
        while True:
            x = random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE
            y = random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
            if (x, y) not in self.snake:
                return (x, y)
    
    def step(self, action):
        new_direction = DIRECTIONS[action]

        # 180 derece dönmeyi engelle
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            new_direction = self.direction  # Eski yönü koru

        self.direction = new_direction
        new_head = (self.snake[0][0] + self.direction[0] * GRID_SIZE, 
                    self.snake[0][1] + self.direction[1] * GRID_SIZE)

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self.spawn_food()
            self.score += 1
            reward = 10
        elif new_head in self.snake or not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
            reward = -10
            return self.get_state(), reward, True
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, False
    
    def get_state(self):
        head = self.snake[0]
        
        # Önündeki 3 yön için duvar veya yılan olup olmadığını kontrol et
        left = (head[0] + DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4][0] * GRID_SIZE,
                head[1] + DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4][1] * GRID_SIZE)
        
        front = (head[0] + self.direction[0] * GRID_SIZE,
                head[1] + self.direction[1] * GRID_SIZE)
        
        right = (head[0] + DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4][0] * GRID_SIZE,
                head[1] + DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4][1] * GRID_SIZE)

        state = [
            # Önünde engel var mı?
            (front in self.snake) or (front[0] < 0 or front[0] >= WIDTH or front[1] < 0 or front[1] >= HEIGHT),
            (right in self.snake) or (right[0] < 0 or right[0] >= WIDTH or right[1] < 0 or right[1] >= HEIGHT),
            (left in self.snake) or (left[0] < 0 or left[0] >= WIDTH or left[1] < 0 or left[1] >= HEIGHT),

            # Yön bilgisi
            self.direction == DIRECTIONS[0],  # Yukarı
            self.direction == DIRECTIONS[1],  # Sağ
            self.direction == DIRECTIONS[2],  # Aşağı
            self.direction == DIRECTIONS[3],  # Sol

            # Yemek konumu
            head[0] < self.food[0], head[0] > self.food[0],
            head[1] < self.food[1], head[1] > self.food[1]
        ]
        return np.array(state, dtype=np.float32)

    
    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], GRID_SIZE, GRID_SIZE))
        pygame.display.flip()
        self.clock.tick(10)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self):
        self.model = DQN(11, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 3)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state_tensor)).item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        batch_state = torch.FloatTensor([b[0] for b in batch])
        batch_action = torch.LongTensor([b[1] for b in batch])
        batch_reward = torch.FloatTensor([b[2] for b in batch])
        batch_next_state = torch.FloatTensor([b[3] for b in batch])
        batch_done = torch.BoolTensor([b[4] for b in batch])

        q_values = self.model(batch_state)
        next_q_values = self.model(batch_next_state).max(1)[0]
        targets = batch_reward + (self.gamma * next_q_values * ~batch_done)

        loss = F.mse_loss(q_values.gather(1, batch_action.unsqueeze(1)).squeeze(), targets)
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    game = SnakeGame()
    agent = Agent()
    episodes = 100
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            game.render()
        print(f"Episode {episode + 1}, Score: {game.score}, Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    train()
