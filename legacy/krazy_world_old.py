from pygame.constants import K_r, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_t, K_a
import pygame
import numpy as np
import sys
from collections import OrderedDict
import os
import random
import copy
clock = pygame.time.Clock()




EASY_GRID_KWARGS = dict(screen_height=256,
                        grid_squares_per_row=5, grid_screens=1,
                        num_goals=3, min_goal_dist=1, max_goal_dist=np.inf,
                        num_keys=0, num_transporters=0,
                        num_steps_before_energy_needed=np.inf, energy_replenish=8, energy_sq_perc=0.00,
                        death_sq_perc=0.15, ice_sq_perc=0.15,
                        dynamics_type='default', reward_type='sparse',
                        render_type='local', headless=False, one_hot_obs=True,
                        seed=None, task_seed=69, init_pos_seed=42)
MEDIUM_GRID_KWARGS = dict(screen_height=256,
                          grid_squares_per_row=7, grid_screens=1,
                          num_goals=1, min_goal_dist=1, max_goal_dist=np.inf,
                          num_keys=0, num_transporters=1,
                          num_steps_before_energy_needed=8, energy_replenish=8, energy_sq_perc=0.05,
                          death_sq_perc=0.1, ice_sq_perc=0.1,
                          dynamics_type='default', reward_type='sparse',
                          render_type='local', headless=False, one_hot_obs=True,
                          seed=None, task_seed=69, init_pos_seed=42)
HARD_GRID_KWARGS = dict(screen_height=256,
                        grid_squares_per_row=10, grid_screens=1,
                        num_goals=3, min_goal_dist=1, max_goal_dist=np.inf,
                        num_keys=1, num_transporters=1,
                        num_steps_before_energy_needed=12, energy_replenish=8, energy_sq_perc=0.1,
                        death_sq_perc=0.15, ice_sq_perc=0.1,
                        dynamics_type='simple', reward_type='sparse',
                        render_type='local', headless=True, one_hot_obs=True,
                        seed=None, task_seed=69, init_pos_seed=42)

from moleskin import moleskin as M


class KrazyGridWorld:
    def seed(self, init_pos_seed, task_seed):
        self.init_pos_rng = random.Random(init_pos_seed)
        self.task_rng = np.random.RandomState(task_seed)
        self.task_rng2 = random.Random(task_seed)

    def __init__(self, screen_height,
                 grid_squares_per_row=10, grid_screens=1,
                 num_goals=3, min_goal_dist=1, max_goal_dist=np.inf,
                 num_keys=1, num_transporters=1,
                 num_steps_before_energy_needed=11, energy_replenish=8, energy_sq_perc=0.00,
                 death_sq_perc=0.07, ice_sq_perc=0.1,
                 dynamics_type='default', reward_type='sparse',
                 render_type='image', headless=False, one_hot_obs=True,
                 seed=42, task_seed=None, init_pos_seed=None):

        # seed itself is depreciated but I'm keeping it at the moment because
        # removing it would break all the other code that depends on this.

        if task_seed is None:
            task_seed = seed

        if init_pos_seed is None:
            init_pos_seed = seed

        self.seed(init_pos_seed, task_seed)

        #seed = task_seed

        #random.seed(seed)
        #np.random.seed(seed)



        if headless is True:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"


        actions = {
            "up": K_UP,
            "left": K_LEFT,
            "right": K_RIGHT,
            "down": K_DOWN,
            'reset': K_r,
            'reset_board': K_t,
            'reset_agent_start': K_a
        }

        self.actions = actions
        self.one_hot_obs = one_hot_obs
        self.screen_dim = (screen_height, screen_height)  # width and height

        #PyGameWrapper.__init__(self, screen_height, screen_height, actions=actions)

        self.render_type = render_type
        self.colors = OrderedDict([('black', (0, 0, 0)),
                                  ('white', (255, 255, 255)),
                                   ("silver", (192, 192, 192)),
                                  ('gold', (255, 223, 0)),
                                  ('green', (0, 255, 0)),
                                  ('brown', (165, 42, 42)),
                                   ('orange', (255, 140, 0)),
                                   ('magenta', (255, 0, 255)),
                                   ('purple', (75, 0, 130)),
                                   ('red', (255, 0, 0))])

        self.tile_types = dict(hole=0, normal=1, goal=2, agent=3, transport=6, door=5, key=7, death=9, ice=8, energy=4)

        self.grid_screens = grid_screens
        self.grid_squares_per_row = grid_squares_per_row
        screen_height = screen_height - 2*0.8*(screen_height//grid_squares_per_row)
        self.grid_square_height = screen_height//grid_squares_per_row
        self.grid_square_margin = int(self.grid_square_height/5)
        self.grid_square_height = int(4*self.grid_square_height/5)
        self.agent_position = None
        self.agent_position_init = None
        self.screen = None
        self.game_grid = None
        self.game_grid_init = None
        self.dynamics = None
        self.num_goals = num_goals
        self.min_goal_dist = min_goal_dist
        self.max_goal_dist = max_goal_dist
        self.num_goals_obtained = 0
        self.goal_squares = None
        self.num_transporters = num_transporters
        self.transporters = None
        self.num_keys = num_keys
        self.has_key = False
        self.dead = False
        self.door_pos = None
        self.energy = num_steps_before_energy_needed
        self.energy_init = num_steps_before_energy_needed
        self.energy_replenish = energy_replenish
        self.death_sq_perc = death_sq_perc
        self.ice_sq_perc = ice_sq_perc
        self.energy_sq_perc = energy_sq_perc
        self.reward_type = reward_type
        self.dynamics_type = dynamics_type
        self.reset_board()
        pygame.init()
        self.screen = pygame.display.set_mode(self.getScreenDims())
        self.clock = pygame.time.Clock()
        self.render()

    def getScreenDims(self):
        return self.screen_dim

    def reset(self):
        # using the same board and the same agent start pos, reset the game.
        self.agent_position = copy.deepcopy(self.agent_position_init)
        self.game_grid = copy.deepcopy(self.game_grid_init)
        self.dead = False
        self.has_key = False
        self.energy = self.energy_init
        self.num_goals_obtained = 0
        self.render()
        return self.get_obs()

    def reset_including_x0(self):
        self.reset_agent_start_position()
        return self.reset()

    def reset_board(self):
        # reset the entire board and agent start position, generating a new MDP.
        self.has_key = False
        self.game_grid = np.ones(dtype=np.int32, shape=(self.grid_screens,
                                                        self.grid_squares_per_row,
                                                        self.grid_squares_per_row))
        self.game_grid *= self.tile_types['normal']
        self.reset_agent_start_position()
        if self.num_keys > 0:
            self.reset_key_position()
        self.reset_transporter_squares()
        self.reset_death_squares()
        self.reset_ice_squares()
        self.reset_energy_squares()
        self.reset_goal_squares()
        #self.change_colors()
        self.change_dynamics()
        self.num_goals_obtained = 0
        self.dead = False
        self.energy = self.energy_init
        self.game_grid_init = copy.deepcopy(self.game_grid)
        self.agent_position = copy.deepcopy(self.agent_position_init)

    def reset_agent_start_position(self):
        # keep the previous board but update the agents starting position.
        # keeps the previous MDP but samples x_0.
        found = False
        while found is False:
            cord_0 = 0
            cord_1 = self.init_pos_rng.randint(0, self.grid_squares_per_row-1)
            cord_2 = self.init_pos_rng.randint(0, self.grid_squares_per_row-1)
            if self.game_grid[cord_0, cord_2, cord_1] == self.tile_types['normal']:
                found = True
                self.agent_position = [cord_0, cord_1, cord_2]
                self.agent_position_init = copy.deepcopy(self.agent_position)
        #self.reset()

        #self.reset()

    def reset_key_position(self):
        def get_corner_square():
            g = self.task_rng.randint(1, self.grid_squares_per_row - 2, (2,))
            if g[0] != self.agent_position[1] or g[1] != self.agent_position[2]:
                return g
            return get_corner_square()
        wall_pos = get_corner_square()

        def get_door_square(wall_pos, range_x, range_y):
            axis = self.task_rng.randint(0, 1)
            if axis == 0:
                door_pos = [self.task_rng.randint(range_x[0], range_x[1]), wall_pos[1]]
            else:
                door_pos = [wall_pos[0], self.task_rng.randint(range_y[0], range_y[1])]
            if door_pos[0] != wall_pos[0] or door_pos[1] != wall_pos[1]:
                return door_pos
            return get_door_square(wall_pos, range_x, range_y)

        def get_key_square_and_agent_square(wall_pos):
            above = self.task_rng.randint(0, 1)
            key_sq = [0, 0]
            agent_sq = [0, 0]
            if above == 0:
                key_sq[0] = self.task_rng.randint(0, wall_pos[0])
                key_sq[1] = self.task_rng.randint(0, wall_pos[1])
                agent_sq[1] = self.task_rng.randint(0, wall_pos[0])
                agent_sq[0] = self.task_rng.randint(0, wall_pos[1])
            else:
                key_sq[0] = self.task_rng.randint(wall_pos[0], self.grid_squares_per_row-1)
                key_sq[1] = self.task_rng.randint(wall_pos[1], self.grid_squares_per_row-1)
                agent_sq[1] = self.task_rng.randint(wall_pos[0], self.grid_squares_per_row - 1)
                agent_sq[0] = self.task_rng.randint(wall_pos[1], self.grid_squares_per_row - 1)
            if agent_sq[1] != key_sq[0] or agent_sq[0] != key_sq[1]:
                if self.game_grid[0, agent_sq[1], agent_sq[0]] == self.tile_types['normal'] and self.game_grid[0, key_sq[1], key_sq[0]] == self.tile_types['normal']:
                    return key_sq, agent_sq
            return get_key_square_and_agent_square(wall_pos)

        if wall_pos[0] > self.grid_squares_per_row // 2:
            range_x = [wall_pos[0], self.grid_squares_per_row - 1]
            for i in range(wall_pos[0], self.grid_squares_per_row):
                self.game_grid[:, i, wall_pos[1]] = self.tile_types['hole']
        else:
            range_x = [0, wall_pos[0]]
            for i in range(0, wall_pos[0]):
                self.game_grid[:, i, wall_pos[1]] = self.tile_types['hole']
        if wall_pos[1] > self.grid_squares_per_row // 2:
            range_y = [wall_pos[1], self.grid_squares_per_row - 1]
            for i in range(wall_pos[1], self.grid_squares_per_row):
                self.game_grid[:, wall_pos[0], i] = self.tile_types['hole']
        else:
            range_y = [0, wall_pos[1]]
            for i in range(0, wall_pos[1]):
                self.game_grid[:, wall_pos[0], i] = self.tile_types['hole']
        door_pos = get_door_square(wall_pos, range_x, range_y)
        self.game_grid[0, door_pos[0], door_pos[1]] = self.tile_types['door']

        key_sq, agent_sq = get_key_square_and_agent_square(wall_pos)
        self.agent_position = [0] + agent_sq
        self.agent_position_init = copy.deepcopy(self.agent_position)
        self.game_grid[0, key_sq[0], key_sq[1]] = self.tile_types['key']
        self.door_pos = [0, door_pos[0], door_pos[1]]

    def get_one_non_agent_square(self):
        g = self.task_rng.randint(0, self.grid_squares_per_row - 1, (2,))
        if g[0] != self.agent_position[1] or g[1] != self.agent_position[2]:
            return g
        return self.get_one_non_agent_square()

    def reset_goal_squares(self):
        gs = []
        self.goal_squares = []
        while len(gs) < self.num_goals:
            g = self.get_one_non_agent_square()
            if self.game_grid[0, g[0], g[1]] == self.tile_types['normal']:
                dist_1 = abs(g[1] - self.agent_position[1])
                dist_2 = abs(g[0] - self.agent_position[2])
                dist = dist_1 + dist_2
                if self.min_goal_dist < dist < self.max_goal_dist:
                    gs.append(g)
        for g in gs:
            self.game_grid[0, g[0], g[1]] = self.tile_types['goal']
            self.goal_squares.append([0, g[0], g[1]])

    def reset_transporter_squares(self):
        gs = []
        for _ in range(self.num_transporters):
            g_1 = self.get_one_non_agent_square()
            g_2 = self.get_one_non_agent_square()
            if self.game_grid[0, g_1[0], g_1[1]] == self.tile_types['normal']:
                if self.game_grid[0, g_2[0], g_2[1]] == self.tile_types['normal']:
                    gs.append([g_1, g_2])
        if len(gs) == self.num_transporters:
            for g in gs:
                for sub_g in g:
                    self.game_grid[0, sub_g[0], sub_g[1]] = self.tile_types['transport']
            self.transporters = gs
        else:
            self.reset_transporter_squares()

    def reset_death_squares(self):
        ds = []
        num_d_squares = int(self.grid_squares_per_row * self.grid_squares_per_row * self.death_sq_perc)
        while len(ds) < num_d_squares:
            d = self.get_one_non_agent_square()
            if self.game_grid[0, d[0], d[1]] == self.tile_types['normal']:
                if self.door_pos is not None:
                    dist_1 = abs(d[0] - self.door_pos[1])
                    dist_2 = abs(d[1] - self.door_pos[2])
                    dist = dist_1 + dist_2
                    if dist > 2:
                        ds.append(d)
                else:
                    ds.append(d)
        for d in ds:
            self.game_grid[0, d[0], d[1]] = self.tile_types['death']

    def reset_ice_squares(self):
        ds = []
        num_d_squares = int(self.grid_squares_per_row * self.grid_squares_per_row * self.ice_sq_perc)
        while len(ds) < num_d_squares:
            d = self.get_one_non_agent_square()
            if self.game_grid[0, d[0], d[1]] == self.tile_types['normal']:
                if self.door_pos is not None:
                    dist_1 = abs(d[0] - self.door_pos[1])
                    dist_2 = abs(d[1] - self.door_pos[2])
                    dist = dist_1 + dist_2
                    if dist > 2:
                        ds.append(d)
                else:
                    ds.append(d)
        for d in ds:
            self.game_grid[0, d[0], d[1]] = self.tile_types['ice']

    def reset_energy_squares(self):
        ds = []
        num_d_squares = int(self.grid_squares_per_row * self.grid_squares_per_row * self.energy_sq_perc)
        while len(ds) < num_d_squares:
            d = self.get_one_non_agent_square()
            if self.game_grid[0, d[0], d[1]] == self.tile_types['normal']:
                ds.append(d)
        for d in ds:
            self.game_grid[0, d[0], d[1]] = self.tile_types['energy']

    def draw_grid(self):
        self.screen.fill(list(self.colors.values())[0])

        # Draw the grid
        for row in range(self.grid_squares_per_row):
            for column in range(self.grid_squares_per_row):
                colour_list = list(self.colors.values())
                colour_idx = self.game_grid[0][row][column]
                color = colour_list[colour_idx]
                pygame.draw.rect(self.screen,
                                 color,
                                 self.get_grid_square_screen_pos(row, column))

    def draw_agent(self):
        colour_list = list(self.colors.values())
        agent_color = (0, 0, 255)  # blue

        pygame.draw.circle(self.screen,
                           agent_color,
                           self.get_agent_screen_position(),
                           self.grid_square_height//2)

    def draw_status(self):
        if self.has_key:
            colour_list = list(self.colors.values())
            color_idx = self.tile_types['key']
            color = colour_list[color_idx]
            pygame.draw.rect(self.screen,
                             color,
                             self.get_grid_square_screen_pos(self.grid_squares_per_row+1, 2))

        if self.energy != np.inf:
            energy_sqs = self.energy // 3
            energy_sqs = int(min(energy_sqs, 5, self.grid_squares_per_row-5))
            for e_step in range(energy_sqs):
                colour_list = list(self.colors.values())
                color_idx = self.tile_types['energy']
                color = colour_list[color_idx]
                pygame.draw.rect(self.screen,
                                 color,
                                 self.get_grid_square_screen_pos(self.grid_squares_per_row+1, 5+e_step))

    def get_grid_square_screen_pos(self, row, column):
        pos = [
            (self.grid_square_margin + self.grid_square_height) * column + self.grid_square_margin,
            (self.grid_square_margin + self.grid_square_height) * row + self.grid_square_margin,
            self.grid_square_height,
            self.grid_square_height
        ]
        return pos

    def get_agent_screen_position(self):
        pos = [
            (self.grid_square_margin + self.grid_square_height) * self.agent_position[1] + self.grid_square_margin + self.grid_square_height // 2,
            (self.grid_square_margin + self.grid_square_height) * self.agent_position[2] + self.grid_square_margin + self.grid_square_height // 2,
        ]
        return pos

    def _handle_player_events(self):
        #  this is bugged with python 3
        #  The pygame support is not very good.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.update_agent_pos('l')

                if key == self.actions["right"]:
                    self.update_agent_pos('r')

                if key == self.actions["up"]:
                    self.update_agent_pos('u')

                if key == self.actions["down"]:
                    self.update_agent_pos('d')

                if key == self.actions['reset']:
                    self.reset()
                if key == self.actions['reset_board']:
                    self.reset_board()
                if key == self.actions['reset_agent_start']:
                    self.reset_including_x0()

    def update_agent_pos(self, command, recurs_step=0):
        if self.dead is False:
            nu_pos = copy.deepcopy(self.agent_position)
            #if command == 'u':
            #    nu_pos[2] = self.agent_position[2] - 1
            #if command == 'd':
            #    nu_pos[2] = self.agent_position[2] + 1
            #if command == 'l':
            #    nu_pos[1] = self.agent_position[1] - 1
            #if command == 'r':
            #    nu_pos[1] = self.agent_position[1] + 1

            nu_pos = self.add_lists(nu_pos, self.dynamics[command])

            if self.is_nu_pos_legal(nu_pos):
                self.agent_position = nu_pos
            self.check_at_goal()
            self.check_at_key()
            self.check_at_transporter()
            self.check_dead()
            self.check_at_energy()
            self.check_at_ice(command, recurs_step=recurs_step)
            if recurs_step == 0:
                self.energy -= 1
            if self.energy < 1:
                self.dead = True

    def is_nu_pos_legal(self, nu_pos):
        if (-1 < nu_pos[1] < self.grid_squares_per_row) and (-1 < nu_pos[2] < self.grid_squares_per_row):  # in bounds
            if self.game_grid[nu_pos[0], nu_pos[2], nu_pos[1]] != self.tile_types['hole']:  # not a hole
                if self.game_grid[nu_pos[0], nu_pos[2], nu_pos[1]] != self.tile_types['door']:
                    return True
                else:
                    if self.has_key:
                        return True
        return False

    def check_at_ice(self, command, recurs_step):
        if recurs_step < 100:
            if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['ice']:
                if command == 'l' or command == 'r':
                    if 0 < self.agent_position[1] < self.grid_squares_per_row - 1:
                        self.update_agent_pos(command, recurs_step=recurs_step+1)
                else:
                    if 0 < self.agent_position[2] < self.grid_squares_per_row - 1:
                        self.update_agent_pos(command, recurs_step=recurs_step+1)

    def check_at_energy(self):
        if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['energy']:
            self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] = self.tile_types['normal']
            self.energy += self.energy_replenish

    def check_at_goal(self):
        if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['goal']:
            self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] = self.tile_types['normal']
            self.num_goals_obtained += 1

    def check_at_key(self):
        if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['key']:
            self.has_key = True
            self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] = self.tile_types['normal']

    def check_at_transporter(self):
        transport_sq = None
        if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['transport']:
            for tr in self.transporters:
                if self.agent_position[2] == tr[0][0] and self.agent_position[1] == tr[0][1]:
                    transport_sq = tr[1]
                elif self.agent_position[2] == tr[1][0] and self.agent_position[1] == tr[1][1]:
                    transport_sq = tr[0]
            if transport_sq is not None:
                self.agent_position = [0, transport_sq[1], transport_sq[0]]

    def check_dead(self):
        if self.game_grid[self.agent_position[0], self.agent_position[2], self.agent_position[1]] == self.tile_types['death']:
            self.dead = True
        #if self.energy == 0:
        #    self.dead = True

    def render(self):
        self.draw_grid()
        self.draw_agent()
        self.draw_status()
        self._handle_player_events()
        self.clock.tick_busy_loop(40000)  # this limits FPS.
        pygame.display.flip()

    def step_rl(self, action):
        action = self.hash_action(action)
        self.update_agent_pos(command=action)
        return self.get_obs(), self.get_reward(), self.dead, dict()

    def step(self, action):
        self.render()
        return self.step_rl(action)

    def hash_action(self, an_int):
        if an_int == 0:
            return 'u'
        elif an_int == 1:
            return 'd'
        elif an_int == 2:
            return 'l'
        elif an_int == 3:
            return 'r'
        else:
            raise NotImplementedError

    def init(self):
        self.reset_board()

    def change_colors(self):
        self.tile_types = dict(zip(list(self.tile_types.keys()), self.task_rng2.sample(list(self.tile_types.values()), len(self.tile_types))))
        return self.tile_types
        #vs = self.tile_types.values()
        #ks = self.tile_types.keys()
        #nu_tile_types = copy.deepcopy(self.tile_types)
        #for
        #items = self.colors.items()
        #random.shuffle(items)
        #self.colors = OrderedDict(items)

    def change_dynamics(self):
        def map_randint_to_movement(randint):
            if randint == 0:
                return [0, 0, -1]
            elif randint == 1:
                return [0, 0, 1]
            elif randint == 2:
                return [0, -1, 0]
            elif randint == 3:
                return [0, 1, 0]

        if self.dynamics_type == 'simple':
            # every button takes one move in a randomized direction.
            random_act = [i for i in range(4)]
            self.task_rng2.shuffle(random_act)
            random_act = [map_randint_to_movement(ra) for ra in random_act]
            self.dynamics = dict(u=random_act[0], d=random_act[1], l=random_act[2], r=random_act[3])
        elif self.dynamics_type == 'moderate':
            # every button takes between one and two moves in a randomized direction.
            random_act = range(4)
            temp = []
            for iter_step in range(2):
                ra = copy.deepcopy(random_act)
                self.task_rng2.shuffle(ra)
                temp.append(copy.deepcopy(ra))
            random_act = temp
            r_act = []
            for rand_actz in random_act:
                r_act.append([map_randint_to_movement(ra) for ra in rand_actz])
            r_act_final = [0 for _ in range(3)]
            r_act_final = [copy.deepcopy(r_act_final) for _ in range(4)]
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        r_act_final[j][k] += r_act[i][j][k]
            random_act = r_act_final
            self.dynamics = dict(u=random_act[0], d=random_act[1], l=random_act[2], r=random_act[3])
        elif self.dynamics_type == 'hard':
            # every button pres takes between one and three moves in a randomized direction.
            random_act = range(4)
            temp = []
            for iter_step in range(3):
                ra = copy.deepcopy(random_act)
                self.task_rng2.shuffle(ra)
                temp.append(copy.deepcopy(ra))
            random_act = temp
            r_act = []
            for rand_actz in random_act:
                r_act.append([map_randint_to_movement(ra) for ra in rand_actz])
            r_act_final = [0 for _ in range(3)]
            r_act_final = [copy.deepcopy(r_act_final) for _ in range(4)]
            for i in range(3):
                for j in range(4):
                    for k in range(3):
                        r_act_final[j][k] += r_act[i][j][k]
            random_act = r_act_final
            self.dynamics = dict(u=random_act[0], d=random_act[1], l=random_act[2], r=random_act[3])
        else:
            self.dynamics = dict(u=[0, 0, -1], d=[0, 0, 1], l=[0, -1, 0], r=[0, 1, 0])
        return self.dynamics

    def add_lists(self, *lists):
        list_final = [0 for _ in range(len(lists[0]))]
        for listz in lists:
            for iter_step, l in enumerate(listz):
                list_final[iter_step] += l
        return list_final

    def get_reward(self):
        if self.reward_type == 'sparse':
            return 0 + self.num_goals_obtained
        else:
            rew = 0
            for goal in self.goal_squares:
                dist_1 = abs(goal[1] - self.agent_position[1])
                dist_2 = abs(goal[2] - self.agent_position[2])
                rew = rew + dist_1 + dist_2
            rew = -1.0*rew
            return rew

    def get_obs(self):
        r_type = self.render_type
        if r_type == 'image':
            return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)
        if r_type == 'global':
            obs = self.game_grid.flatten(), np.array(self.agent_position[1:])
        elif r_type == 'local':
            neighbors = []
            v, x, y = self.agent_position
            for _i, _j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]:
                i, j = (_i + x, _j + y)
                if 0 <= i < self.grid_squares_per_row and 0 <= j < self.grid_squares_per_row:
                    neighbors.append(self.game_grid[0, j, i])
                else:
                    neighbors.append(None)
            obs = np.array(neighbors + [v]), np.array([x, y])
        if self.one_hot_obs:
            x, y = obs[1]
            obs = self.one_hot(obs[0], len(self.tile_types)), \
                  self.one_hot(np.array([x * self.grid_squares_per_row + y]), self.grid_squares_per_row ** 2)

        return np.concatenate(list(map(lambda o: o.flatten(), obs)))

    @staticmethod
    def one_hot(vec, size):
        flattened = vec.flatten()
        state_len = flattened.shape[0]
        oh = np.zeros((state_len, size))
        for i, s in enumerate(flattened):
            oh[i][s] = 1
        return oh


def run_grid():
    game = KrazyGridWorld(**MEDIUM_GRID_KWARGS)
    game.change_colors()
    game.reset_including_x0()

    import random
    import time
    while True:
        game.render()
        for i in range(3):
                time.sleep(0.5)
                rint = random.randint(0, 3)
                game.step(rint)
                game.render()

        time.sleep(1.5)
        game.reset_board()
        game.change_colors()

if __name__ == "__main__":
    run_grid()