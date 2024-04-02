import math

import pygame
from enum import Enum

import torch.nn
import numpy as np

BLACK = (0, 0, 0)
White = (255, 255, 255)
Red = (255, 0, 0)
width = 900
height = 900


#		state = [self.king.x, self.king.y]

class NETWORK(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(NETWORK, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x


class DDQN(object):
    def __init__(self):
        self.target_net = NETWORK(2, 4, 32)
        self.eval_net = NETWORK(2, 4, 32)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        self.memory_counter = 0
        self.memory_size = 50000
        self.memory = np.zeros((self.memory_size, 7))
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.alpha = 0.99
        self.batch_size = 64
        self.episode_counter = 0
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.dx=0

    def memory_store(self, s0, a0, r, s1, sign):
        transition = np.concatenate((s0, [a0, r], s1, [sign]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, states: np.ndarray) -> int:
        state = torch.unsqueeze(torch.tensor(states).float(), 0)
        if np.random.uniform() > self.epsilon:
            logit = self.eval_net(state)
            action = torch.argmax(logit, 1).item()
        else:
            action = int(np.random.choice(4, 1))
        return action

    def policy(self, states: np.ndarray) -> int:
        state = torch.unsqueeze(torch.tensor(states).float(), 0)
        logit = self.eval_net(state)
        action = torch.argmax(logit, 1).item()
        return action

    def train(self, s0, a0, r, s1, sign):
        if sign == 1:
            if self.episode_counter % 2 == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.episode_counter += 1
        self.memory_store(s0, a0, r, s1, sign)
        self.epsilon = np.clip(self.epsilon * self.epsilon_decay, a_min=0.01, a_max=None)

        if self.memory_counter > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[batch_index]
        # state has 2 values/(maybe change it to 4 later)
        batch_s0 = torch.tensor(batch_memory[:, :2]).float()
        batch_a0 = torch.tensor(batch_memory[:, 2:3]).float()
        batch_r = torch.tensor(batch_memory[:, 3:4]).float()
        batch_s1 = torch.tensor(batch_memory[:, 4:6]).float()
        batch_sign = torch.tensor(batch_memory[:, 6:7]).float()
        q_eval = self.eval_net(batch_s0).gather(1, batch_a0.long())

        with torch.no_grad():
            maxAction = torch.argmax(self.eval_net(batch_s1), 1, keepdim=True)
            q_target = batch_r + (1 - batch_sign) * self.alpha * self.target_net(batch_s1).gather(1, maxAction)

        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP_RIGHT = 3
    UP_LEFT = 4


class Game:
    def set_text(self, string, coordx, coordy, fontSize):  # Function to set text

        font = pygame.font.Font('04B_30__.TTF', fontSize)
        # (0, 0, 0) is black, to make black text
        text = font.render(string, True, Red)
        textRect = text.get_rect()
        textRect.center = (coordx, coordy)
        return (text, textRect)

    def __init__(self, max_step=float('inf')):
        self.Tile = None
        self.w = width
        self.h = height
        self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)
        pygame.init()
        self.clock = pygame.time.Clock()
        self.step_counter = 0
        self.max_step = max_step
        self.visited = {}
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('JumpGame')
        self.clock = pygame.time.Clock()
        self.ListOfThem = [
            pygame.Rect(260, 550, 100, 20),  # 1
            pygame.Rect(500, 600, 100, 20),  # 2
            pygame.Rect(750, 500, 100, 20),  # 3
            pygame.Rect(20, 460, 100, 20),  # 4
            pygame.Rect(300, 310, 100, 20),  # 5
            pygame.Rect(550, 310, 100, 20),  # 6
            pygame.Rect(0, 200, 100, 20),  # 7
            pygame.Rect(800, 150, 100, 20),  # 8
            pygame.Rect(0, self.h - 140, self.w, 10)  # Platform 11
        ]
        self.ListofListofThem = self.ListOfThem.copy()
        self.reset()
        self.char = pygame.image.load("img_2.png").convert()
        self.char = pygame.transform.scale(self.char, (60, 60))
        self.x = pygame.image.load("430306741_263134090199334_349620565060820910_n.png").convert()
        self.x = pygame.transform.scale(self.x, (100, 20))
        self.background = pygame.image.load("img_3.png").convert()
        self.background = pygame.transform.scale(self.background,(900,900))
        self.velocity = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 15
        self.game_over = False

    def draw_text(self, text, text_col, x, y):
        font = pygame.font.Font("04B_30__.TTF", 24)
        img = font.render(text, True, text_col)
        self.display.blit(img, (x, y))

    def Collisions(self, Tile):
        if (self.D.x < Tile.x + Tile.w and
                self.D.x + self.D.w > Tile.x and
                self.D.y < Tile.y + 1.5 * Tile.h and
                self.D.y + self.D.h > Tile.y):
            self.velocity = 0

            self.is_jumping = True

            if (Tile.y - self.D.y < self.D.h) and not (Tile.x - 52 < self.D.x):
                self.D.x = Tile.x - self.D.w


            elif (Tile.y - self.D.y < self.D.h) and not (Tile.x > self.D.x - Tile.w + 8):
                self.D.x = Tile.x + Tile.w

            elif (self.D.y + self.D.h > Tile.y) and (self.D.y < Tile.y):
                self.Tile = Tile
                self.D.y = Tile.y - self.D.h
                self.is_jumping = False
            else:
                self.on_ground = False
                if self.D.y > self.h - 200:
                    self.velocity += 0.5  # Increase velocity due to gravity
                    self.D.y += self.velocity

    def reset(self):
        self.step_counter = 0
        done = False
        state = [self.D.x, self.D.y]
        self.visited = {}
        self.visited[(self.D.y)] = 1
        self.direction = None
        self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)
        self.frame_iteration = 0
        self.ListofListofThem = self.ListOfThem.copy()
        self.Tile = None
        return done, state

    # MOOD
    def play_step(self, action):
        old_y = self.D.y
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)

        self.clock.tick(60)
        while True:
            self._update_ui()
            self.step_counter += 1
            state = [self.D.x, self.D.y]
            if self.D.y > height:
                self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)

            if self.D.y > old_y:
                reward = 0

            else:

                self.visited[(self.D.y)] = self.visited.get((self.D.y), 0) + 1
                try:
                    if self.visited[self.D.y] < self.visited[old_y]:
                        self.visited[self.D.y] = self.visited[old_y] + 1
                except:
                    pass

                reward = -self.visited[self.D.y]

            done = True if self.step_counter > self.max_step else False
            return state, reward, done

    # MOOD
    def _move(self, action):
        clock_wise = [Direction.UP_RIGHT, Direction.UP_LEFT, Direction.RIGHT, Direction.LEFT]

        self.direction = clock_wise[action]
        if self.direction == Direction.UP_RIGHT and not self.is_jumping:
            self.D.x += 8
            self.dx=8
            self.is_jumping = True
            self.velocity = -self.jump_height
            self.on_ground = False
        elif self.direction == Direction.UP_LEFT and not self.is_jumping:
            self.D.x -= 8
            self.dx=-8

            self.is_jumping = True
            self.velocity = -self.jump_height
            self.on_ground = False

        elif self.direction == Direction.RIGHT :
            if not self.is_jumping:
                self.D.x += 8


        elif self.direction == Direction.LEFT:
            if not self.is_jumping:
                self.D.x -= 8

        if not self.on_ground:
            if self.is_jumping:
                self.D.x+=self.dx
            self.velocity += 0.5
            self.D.y += self.velocity


        # Apply gravity
        if not self.on_ground:
            self.velocity += 0.5  # Increase velocity due to gravity
            self.D.y += self.velocity
            for i in self.ListOfThem:
                self.Collisions(i)

        # Keep the character within the game boundaries
        if self.D.x < 0:
            self.D.x = 0
        elif self.D.x > self.w - 60:
            self.D.x = self.w - 60

    def _update_ui(self):
        self.display.fill(BLACK)
        self.display.blit(self.background, (0, 0))
        for i in self.ListOfThem:
            pygame.draw.rect(self.display, White, i)
        for i in range(len(self.ListOfThem)-1):
            self.display.blit(self.x, (self.ListOfThem[i].x, self.ListOfThem[i].y))
        Show = self.set_text("MOTHER FUCKER", self.w / 2, 40, 20)
        self.display.blit(Show[0], Show[1])

        pygame.draw.rect(self.display, White, self.D)
        self.display.blit(self.char, (self.D.x, self.D.y))

        pygame.display.flip()


def train():
    agent = DDQN()
    env = Game(max_step=1000)
    num_episode = 100000
    for i in range(num_episode):
        done, state = env.reset()
        running_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.play_step(action)
            running_reward += reward
            sign = 1 if done else 0
            agent.train(state, action, reward, next_state, sign)
            state = next_state
        print(f'episode: {i}, reward: {running_reward}')


if __name__ == '__main__':
    train()