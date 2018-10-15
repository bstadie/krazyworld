# This file implements the following contrived GRID In which exploration is supposed to be hard??
#  ------------------------
#  |xxxxxxTxxxxxxx|
#  |xxxxxxOxxxxxxx|
#  |xxxxxxOxxxxxxx|
#  |xxxxxxOxxxxxxx|
#  |xxxxxxOxxxxxxx|
#  |SOOOOOOOOOOOOO|

import numpy as np
import copy
import random
import gym
from gym import spaces
from gym.utils import seeding
import cv2



WIDTH = 14
HEIGHT = WIDTH // 2
IMG_SCALE = 20  # img is n times larger (pixels) than width/height.
#IMG_SIZE_W = WIDTH*6
#IMG_SIZE_H = IMG_SIZE_W // 2


class ContrivedGrid(gym.Env):
    def __init__(self, img_obs=False):
        self.img_obs = img_obs
        if self.img_obs is False:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*WIDTH*HEIGHT,))
        else:
            self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(HEIGHT, WIDTH, 3))
        self.action_space = spaces.Discrete(4)
        grid = np.zeros((HEIGHT, WIDTH))
        grid[0][:] = 1
        for i in range(HEIGHT-1):
            grid[i][WIDTH//2] = 1
        grid[HEIGHT-1][WIDTH//2] = 2
        self.goal_yx = [HEIGHT-1, WIDTH//2]
        self.grid = grid
        self.agent_pos_basis = np.zeros((HEIGHT, WIDTH))
        self.agent_pos_yx = [0, 0]
        self.agent_pos_basis[0][0] = 1
        self.reset()
        self.seed()
        self.viewer = None
        self.state = None
        self.max_horizon = 50
        self.current_step = 0

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.current_step += 1
        self.apply_action(action)
        obs = self.get_obs()
        rew = self.get_reward()
        done = self.get_done()
        if self.current_step == self.max_horizon:
            done = True
        #print(done)
        return obs, rew, done, dict(ground_truth_yx=self.agent_pos_yx)

    def reset(self):
        self.agent_pos_yx = [0, 0]
        self.agent_pos_basis[0][0] = 1
        self.current_step = 0
        return self.get_obs()

    def render(self, mode='human'):
        pass

    def get_done(self):
        #print(int(self.grid[self.agent_pos_yx[0]][self.agent_pos_yx[1]]))
        if int(self.grid[self.agent_pos_yx[0]][self.agent_pos_yx[1]]) == 0:
            return True
        if int(self.grid[self.agent_pos_yx[0]][self.agent_pos_yx[1]]) == 2:
            return True
        return False

    def apply_action(self, action):
        action_idx = action #np.argmax(action)
        if action_idx == 0:  # right
            self.agent_pos_yx[1] = min(WIDTH-1, self.agent_pos_yx[1]+1)
        elif action_idx == 1:  # left
            self.agent_pos_yx[1] = max(0, self.agent_pos_yx[1] - 1)
        elif action_idx == 2:  # up
            self.agent_pos_yx[0] = min(HEIGHT-1, self.agent_pos_yx[0] + 1)
        else:  # down
            self.agent_pos_yx[0] = max(0, self.agent_pos_yx[0] - 1)
        self.agent_pos_basis = np.zeros((HEIGHT, WIDTH))
        self.agent_pos_basis[self.agent_pos_yx[0], self.agent_pos_yx[1]] = 1

    def get_obs(self):
        if self.img_obs is False:
            full_obs = np.concatenate((copy.deepcopy(self.grid),
                                       copy.deepcopy(self.agent_pos_basis)))
            return full_obs.flatten()
        else:
            fake_img = np.zeros((HEIGHT, WIDTH, 3))

            one_idxs = self.grid == 1 #.transpose((1, 0)) == 1
            one_idxs = one_idxs.astype(int)
            one_idxs = np.tile(np.expand_dims(one_idxs, -1), 3)
            one_idxs = one_idxs*np.array([255, 0, 0])
            fake_img += one_idxs

            two_idxs = self.grid == 2 #.transpose((1, 0)) == 2
            two_idxs = two_idxs.astype(int)
            two_idxs = np.tile(np.expand_dims(two_idxs, -1), 3)
            two_idxs *= np.array([0, 255, 0])
            fake_img += two_idxs

            #agent_idx = self.agent_pos_basis == 1 #.transpose((1, 0)) == 1
            #agent_idx = agent_idx.astype(int) * 255
            #fake_img[:, :, 2] = agent_idx
            fake_img[self.agent_pos_yx[0], self.agent_pos_yx[1], :] = np.array([0, 0, 255])

            show = False
            res = fake_img
            if show:
                #res = fake_img
                #res = cv2.resize(fake_img, dsize=(IMG_SCALE*WIDTH, IMG_SCALE*HEIGHT), interpolation=cv2.INTER_NEAREST)
                #res = cv2.circle(fake_img, (10, 2), 1, color=(255, 255, 255))
                fake_goal_y = np.random.randint(0, HEIGHT-1)
                fake_goal_x = np.random.randint(0, WIDTH-1)
                #fake_img[self.agent_pos_yx[0], self.agent_pos_yx[1], :] = 255
                fake_img[fake_goal_y, fake_goal_x, :] = 255
                res = cv2.resize(fake_img, dsize=(IMG_SCALE * WIDTH, IMG_SCALE * HEIGHT), interpolation=cv2.INTER_NEAREST)
                res = res.astype(np.uint8)
                #res = cv2.resize(res, dsize=(IMG_SCALE * WIDTH, IMG_SCALE * HEIGHT), interpolation=cv2.INTER_NEAREST)

                from PIL import Image

                img = Image.fromarray(res, 'RGB')
                img.show()

            return res

            #data = np.zeros((h, w, 3), dtype=np.uint8)
            #data[256, 256] = [255, 0, 0]


            #fake_img_scaled = np.zeros((WIDTH*IMG_SCALE, HEIGHT*IMG_SCALE, 3))
            #for _ in range(IMG_SCALE):
            #    fake_img_scaled[]
            #return fake_img

            #for i in range(0, WIDTH):
            #    for j in range(0, HEIGHT):
            #        if self.grid[j][i] == 1:
            #            fake_img[j][i][0] = 255
            #        elif self.grid[j][i] == 2:
            #            fake_img[j][i][1] = 255

    def get_reward(self):
        if self.grid[self.agent_pos_yx[0]][self.agent_pos_yx[1]] == 2:
            return 1
        return 0


def main():
    grid = ContrivedGrid()
    #ob, rew, done, _ = grid.step(2)
    #grid.reset()
    grid = ContrivedGrid(img_obs=True)
    #ob, rew, done, _ = grid.step(2)
    #grid.reset()
    solution = [
                #[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                #[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]
                ]
    for _ in range(WIDTH//2):
        solution.append(0)
    for _ in range(HEIGHT-1):
        solution.append(2)
    d = grid.get_done()
    #a = grid.grid.transpose((1, 0))
    if d is True:
        assert False
    for s in solution:
        ob, rew, done, _ = grid.step(s)
        if done:
            print(grid.agent_pos_yx)
            print(grid.agent_pos_basis)
            print(grid.grid)
            if rew != 1:
                assert False
    print(rew)

    #for _ in range(10000):
    #    idx = random.randint(0, 3)
        #act = [0, 0, 0, 0]
        #act[idx] = 1
    #    grid.step(idx)

if __name__ == '__main__':
    main()
