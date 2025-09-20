# -*- coding:utf-8 -*-
"""
作者：王红梅
日期：2023年11月23日
有虚拟leader，采取课程分步训练，两个阶段：导航和组队，第一阶段训练完毕后，采用此环境，加入了阶段判断，重点是编队的训练。
"""
import gym
import matplotlib
from gym import spaces
import random

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib import animation
import csv
import imageio
from lidar import Lidar
import argparse
from formation import Formation
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--max_episode_steps', type=int, default=300)
parser.add_argument('--pursuer_isTotal_range', type=bool, default=False)
parser.add_argument('--pursuer_lidar_lines', type=int, default=25)
parser.add_argument('--pursuer_safe_radius', type=float, default=10.0) #安全半径
parser.add_argument('--pursuer_radius_thr', type=float, default=20.0) #雷达测量半径
parser.add_argument('--evader_isTotal_range', type=bool, default=True)
parser.add_argument('--evader_lidar_lines', type=int, default=25)
parser.add_argument('--evader_radius_thr', type=float, default=10.0)
parser.add_argument('--evader_move', type=bool, default=True)
parser.add_argument('--velocity_bound', type=float, default=0.5)
parser.add_argument('--angle_bound', type=float, default=np.pi / 6)
parser.add_argument('--max_velocity', type=float, default=2)
parser.add_argument('--evader_xy_speed', type=float, default=1)

args = parser.parse_args()

# env_param_dict = {
#     'width': 300,
#     'height': 300,
#     # 'obstacle_params': [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
#     #                     (175, 75, 8), (175, 135, 8), (200, 100, 5), (225, 125, 5), (200, 85, 5),
#     #                     (225, 170, 5), (200, 150, 5), (125, 100, 8), (225, 200, 5), (125, 175, 5),
#     #                     (75, 100, 8), (100, 200, 5), (200, 150, 5), (100, 110, 8), (200, 170, 5), (160, 230, 8),
#     #                     (60, 70, 10)]
#     'obstacle_params': [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
#                         (175, 75, 8), (175, 135, 8), (200, 100, 5), (225, 125, 5), (200, 85, 5),
#                         (225, 170, 5), (200, 150, 5)]
# }


class Environment_2D(gym.Env):

    """
	环境：二维，连续动作，连续状态
	目的：多个无人机 避障，防撞 追踪同一运动目标
	动作：角度，变化范围（0, 2*pi）；速度，变化范围（0，2），均为增量式
	状态：相对位置信息+环境障碍物信息（不将其他uav看做障碍物，为下一步队形做准备）
	step：分为单个agent步进 single_step方法， 多个agent同时步进 step方法
	reset：产生所有agent状态
	render：画出所有无人机
	"""

    def __init__(self, width=100, height=100,disturbance = 0,purnum=4): #4个中有一个是虚拟领航
        super(Environment_2D, self).__init__()
        # self.obstacle_params=[(75, 175, 5), (100, 125, 5), (125, 225, 5), (175, 200, 10),
        #                 (175, 75, 3), (175, 135, 3), (200, 100, 5), (125, 125, 5), (50, 85, 5),
        #                 (25, 170, 5), (70, 150, 5)]
        self.width =width
        self.obstacles=[]
        self.disturbance = disturbance
        self.purnum = purnum
        self.possible_agents = list(range(0, self.purnum))
        self.agents = list(range(0, self.purnum))
        self.param_set(width, height)
        self.pursuer_radius = 2
        self.pursuer_pos_x = np.full((self.purnum), 0)
        self.pursuer_pos_y = np.full((self.purnum), 0)
        self.create_evader()
        self.create_pursuer()
        # self.create_random_obstacles(1)
        self.create_random_obstacles(5) #固定位置的障碍物
        self.create_fig_ax()
        self.evader_lidar = Lidar(lidar_lines=args.evader_lidar_lines, isTotal_range=args.evader_isTotal_range,
                                  radius_thr=args.evader_radius_thr)
        self.pursuer_lidar = Lidar(lidar_lines=args.pursuer_lidar_lines, isTotal_range=args.pursuer_isTotal_range,
                                   radius_thr=args.pursuer_radius_thr)
        self.pursuer_effective_range = self.radius_thr - self.pursuer_radius # 有效探测距离
        self.K_DISTANCE = 0.5
        self.K_THETA = 0.2
        self.K_CRASH = 2
        self.K_OBSTACLE = self.pursuer_effective_range / 10
        # self.K_OBSTACLE = self.pursuer_effective_range / 10
        self.episode_steps = 0
        # 定义动作空间，值为【-1,1】区间的含有2个1维元素的
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.whole_action_space = spaces.Box(low=-1, high=1, shape=(2 * self.purnum,), dtype=np.float32)
        # 定义状态空间，20个1维元素

        # self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        # 增加其它agent的位置，vileader的位置，与其它agent的距离，与virleader的距离
        shape = args.evader_lidar_lines-2 + 11 + self.purnum * 2
        virleader_shape = args.evader_lidar_lines-2 + 11
        self.observation_space = spaces.Box(low=-1, high=1, shape=(shape,), dtype=np.float32)
        self.virleader_observation_space = spaces.Box(low=-1, high=1, shape=(virleader_shape,), dtype=np.float32)#雷达线增至16条
        # self.whole_observation_space = spaces.Box(low=-1, high=1, shape=(20 * self.purnum,), dtype=np.float32)
        self.whole_observation_space = spaces.Box(low=-1, high=1, shape=(shape * (self.purnum-1)+virleader_shape,), dtype=np.float32)
        self.scan_info = np.full((self.purnum, self.lidar_lines), self.radius_thr, dtype=float)
        self.last_scan_info = self.scan_info.copy()
    def param_set(self, width, height):
        """环境中相关超参数的初始化"""
        self.width = width
        self.height = height
        self.lidar_lines = args.pursuer_lidar_lines
        self.evader_speed = np.array([-1, -1]) # 设置固定速度，如果要evader是随机速度，可以在evader_random_reset里修改参数
        self.radius_thr = 20
        self.capture_distance = 5
        self.nActions = 2
        self.nStates_pos = 11 + self.purnum * 2
        self.nStates_obs = self.lidar_lines - 2
        self.nStates = self.nStates_pos + self.nStates_obs  # Remove the 0 degree and 180 degree rays
        self.cum_distance_history = 0
        self.angle_bound = args.angle_bound
        self.velocity_bound = args.velocity_bound
        # self.obstacle_params = [(25, 75, 8), (50, 125, 5), (75, 25, 5), (100, 100, 10), (125, 50, 8), (150, 175, 5),
        # 						(175, 50, 5), (150, 100, 5), (75, 175, 8), (175, 150, 5), (20, 20, 5)]  # (center_x, center_y, radius)
        # self.obstacle_params = [(25, 125, 8), (50, 75, 5), (75, 175, 5), (125, 150, 10), (125, 25, 8), (150, 100, 5),
        # 						(175, 75, 5), (150, 100, 5), (75, 50, 8),
        # 						(175, 150, 5), (75, 125, 5)]  # (center_x, center_y, radius)
        # self.obstacle_params = [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
        #                         (175, 75, 8), (200, 150, 5), (225, 125, 5), (200, 150, 5),
        #                         (125, 100, 8), (225, 200, 5), (125, 175, 5)]  # (center_x, center_y, radius)
        # self.obstacle_params = [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
        #                         (175, 75, 8), (175, 135, 8), (200, 100, 5), (225, 125, 5), (200, 85, 5),
        #                         (225, 170, 5), (200, 150, 5), (125, 100, 8), (225, 200, 5), (125, 175, 5),
        #                         (75, 100, 8), (100, 200, 5), (200, 150, 5), (100, 110, 8), (200, 170, 5), (160, 230, 8),
        #                         (60, 70, 10)]
        # self.pursuer_growing_pos = [(100, 85), (100, 100), (100, 115)]
        # self.growing_pos_index = 0
        # self.obstacles = []
        self.isSuccessful = np.full((self.purnum),False)
        self.current_distance = [0] * self.purnum

        self.iscollision = np.full((self.purnum), False)

    def pdf(self, mu, sigma, x):
        """标准正态分布"""
        denominator = np.sqrt(2 * np.pi) * sigma
        return np.exp(-0.5 * np.square((x - mu) / sigma)) / denominator

    def sign(self, x, y):
        """符号函数"""
        if x > y:
            return 1
        else:
            return -1

    def distance_2d(self, x1, y1, x2, y2):
        """二维平面距离"""
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    def angel_diff_2d(self, x1, y1, x2, y2, alpha):
        """二维平面角度差，（x1, y1）到（x2, y2）向量 与 alpha 的角度差"""
        direction_vec = np.array([np.cos(alpha), np.sin(alpha)])
        eva_pur_vec = np.array([x2 - x1, y2 - y1])
        angle_diff = np.arccos(
            np.sum(eva_pur_vec * direction_vec) / (np.linalg.norm(eva_pur_vec) + 1e-6))  # angel_diff belong to (0, pi)

        return angle_diff

    def get_true_action(self, action):  # [angle, velocity], belong to (-1, 1)
        """根据网络输出的动作转化为环境实际的动作"""
        action = np.clip(action, -1, 1)
        # angle
        action[0] = action[0] * self.angle_bound #动作方向范围-30,30度
        # velocity
        action[1] = action[1] * self.velocity_bound# 速度范围 -0.5,0.5

        return action

    def pos_to_state(self, x1, y1, x2, y2, alpha, num):
        """根据智能体观测的信息组合成状态信息，追踪者，逃避者位置信息，与障碍物距离信息,alpha"""
        # --------------------------------------------------------------------------
        # 获取当前时刻追踪和逃避者的相对位置信息
        # --------------------------------------------------------------------------
        relative_distance_x = (x2 - x1) / self.width  # belong to (-1, 1)
        relative_distance_y = (y2 - y1) / self.height  # belong to (-1, 1)
        relative_distance = np.sqrt(
            (np.square(relative_distance_x) + np.square(relative_distance_y)) / 2)  # belong to (0, 1)
        relative_distance_norm = (relative_distance - 0.5) * 2  # belong to (-1, 1)
        relative_angle_norm = np.arctan2(y2 - y1, x2 - x1) / np.pi  # belong to (-1, 1)

        # if num == self.purnum-1:  # 获取虚拟leader调整后的位置及速度信息
        #     pursuer_x_norm = ((self.pursuer_pos_x[num] / self.width) - 0.5) * 2  # belong to (-1, 1)
        #     pursuer_y_norm = ((self.pursuer_pos_y[num] / self.height) - 0.5) * 2  # belong to (-1, 1)
        #     pursuer_angle_norm = (self.pursuer_angle[num] / np.pi) - 1  # belong to (-1, 1)
        #     pursuer_vel_norm = (2 * (self.pursuer_velocity[num] / args.max_velocity)) - 1
        # else:
        # 获取所有追踪者信息,包括virleader
        pursuer_x_norm = ((self.pursuer_pos_x / self.width) - 0.5) * 2  # belong to (-1, 1)
        pursuer_y_norm = ((self.pursuer_pos_y / self.height) - 0.5) * 2  # belong to (-1, 1)
        pursuer_angle_norm = (self.pursuer_angle / np.pi) - 1  # belong to (-1, 1)
        pursuer_vel_norm = (2 * (self.pursuer_velocity / args.max_velocity)) - 1

        # 获取逃避者调整后的位置及速度信息
        evader_x_norm = ((self.evader_pos_x / self.width) - 0.5) * 2  # belong to (-1, 1)
        evader_y_norm = ((self.evader_pos_y / self.height) - 0.5) * 2  # belong to (-1, 1)
        evader_angle_norm = (self.evader_angle / np.pi) - 1  # belong to (-1, 1)
        evader_vel = np.array(self.evader_speed) / args.evader_xy_speed
        evader_vel_norm = (2 * np.sqrt(
            (np.square(evader_vel[0]) + np.square(evader_vel[1])) / 2)) - 1  # belong to (-1, 1)
        # 获取角度差信息
        angle_diff = self.angel_diff_2d(x1, y1, x2, y2, alpha)
        angle_diff_norm = ((angle_diff / np.pi) - 0.5) * 2  # 将角度截断在(-1, 1)之间，便于网络处理
        # 合成位置状态信息
        if num != self.purnum-1:
            s_pos = np.hstack((pursuer_x_norm, pursuer_y_norm, pursuer_x_norm[num], pursuer_y_norm[num],pursuer_angle_norm[num], pursuer_vel_norm[num],
                           evader_x_norm, evader_y_norm, evader_angle_norm, evader_vel_norm,
                           relative_angle_norm, relative_distance_norm, angle_diff_norm))
        else:
            s_pos = np.hstack((pursuer_x_norm[num], pursuer_y_norm[num],pursuer_angle_norm[num], pursuer_vel_norm[num],
                           evader_x_norm, evader_y_norm, evader_angle_norm, evader_vel_norm,
                           relative_angle_norm, relative_distance_norm, angle_diff_norm))
        # ---------------------------------------------------------------------------
        # 获取周围环境信息
        # ---------------------------------------------------------------------------
        self.scan_info[num,:] = self.pursuer_lidar.scan_to_state(pos_x=self.pursuer_pos_x[num], pos_y=self.pursuer_pos_y[num],
                                                          angle=self.pursuer_angle[num],
                                                          obstacle_params=self.obstacle_params) - self.pursuer_radius
        scan_info = []
        scan_info = self.scan_info[num,1:-1]  # 去掉了第一个和最后一个信息
        s_scan = scan_info / self.K_OBSTACLE
        s_scan_norm = (s_scan - 0.5) * 2  # 将环境信息截断在(-1, 1)之间，便于网络处理
        # 合成综合状态信息
        s = np.hstack((s_pos, s_scan_norm))
        return s


    def evader_random_reset(self, speed_random=True):
        """逃避者随机初始化"""
        self.evader_pos_x = np.random.uniform(0.3*self.width, 0.8*self.width)
        self.evader_pos_y = np.random.uniform(0.3*self.width, 0.8*self.width)
        if speed_random:
            self.evader_speed = np.random.uniform(-args.evader_xy_speed, args.evader_xy_speed, 2)
        self.evader_angle = np.arctan2(self.evader_speed[1], self.evader_speed[0])
        self.evader_angle = self.angle_normalize(self.evader_angle) #belong to (0,2*pi)

    def multi_pursuer_random_reset(self):
            """追踪者初始位置随意，但相近，虚拟领航初始位置在中心"""
            if np.random.uniform() > 0.5:
                self.pursuer_pos_x[0] = np.random.uniform(0, 0.5*self.width)
            else:
                self.pursuer_pos_x[0] = np.random.uniform(0.5*self.width, 0.9*self.width)
            if np.random.uniform() > 0.5:
                self.pursuer_pos_y[0] = np.random.uniform(0, 0.5*self.width)
            else:
                self.pursuer_pos_y[0] = np.random.uniform(0.5*self.width, 0.9*self.width)
            if self.purnum>2:
                for i in range(self.purnum):
                    if i>0 and i!=self.purnum-1:
                        self.pursuer_pos_x[i] = self.pursuer_pos_x[0] + np.random.uniform(10,self.pursuer_radius)
                        self.pursuer_pos_y[i] = self.pursuer_pos_y[0] + np.random.uniform(10,self.pursuer_radius)

            self.pursuer_pos_x[-1] = np.average(self.pursuer_pos_x[:(self.purnum-1)])
            self.pursuer_pos_y[-1] = np.average(self.pursuer_pos_y[:(self.purnum-1)])
            # self.pursuer_angle = [3.14,3.14,3.14]
            # self.pursuer_velocity = [1,2,2]
            self.pursuer_angle = np.random.uniform(0, 2 * np.pi, self.purnum)
            self.pursuer_velocity = np.random.uniform(0, args.max_velocity, self.purnum)
    def create_evader(self):
        """创造逃避者"""
        self.evader_random_reset()
        self.evader_radius = 2
        self.evader = patches.Circle((self.evader_pos_x, self.evader_pos_y), radius=self.evader_radius, fc='r')

        # self.evader_traj = patches.Circle((self.evader_pos_x, self.evader_pos_y), radius=self.evader_radius, fc='g')

    def create_pursuer(self):
        """创造追踪者"""
        self.multi_pursuer_random_reset()
        self.pursuer_radius = 2
        self.pursuer=[]
        self.pursuer_field=[]
        for i in range(self.purnum):
            pursuer_circle = patches.Circle((self.pursuer_pos_x[i], self.pursuer_pos_y[i]), radius=self.pursuer_radius, fc='g')
            self.pursuer.append(pursuer_circle)
            field_circle = patches.Circle((self.pursuer_pos_x[i], self.pursuer_pos_y[i]),radius=self.radius_thr, fc=None, ec='k', alpha=0.3)  # 雷达范围
            self.pursuer_field.append(field_circle)


    def create_obstacles(self):
        """创造障碍物，固定位置和大小的障碍物"""
        for obstacle_param in self.obstacle_params:
            self.obstacles.append(patches.Circle((obstacle_param[0], obstacle_param[1]),
                                                 radius=obstacle_param[2], fc='k'))

    def create_random_obstacles(self, num):
        """创造随机位置和大小障碍物"""
        # if np.random.uniform() > 0.5:
        #     x = np.random.uniform(75, 125, num)
        # else:
        #     x = np.random.uniform(175, 225, num)
        # if np.random.uniform() > 0.5:
        #     y = np.random.uniform(75, 125, num)
        # else:
        #     y = np.random.uniform(175, 225, num)
        self.obstacles = []
        self.obstacle_params = []
        x = np.random.uniform(0, self.width-10, num)
        y = np.random.uniform(0, self.width-10, num)
        r = np.random.uniform(3,8, num)
        for i in range(num):
            self.obstacles.append(patches.Circle((x[i], y[i]), radius=r[i], fc='k'))
            self.obstacle_params.append((x[i],y[i],r[i]))

    def collision_detect(self,num):
        """碰撞检测"""
        for obstacle in self.obstacle_params:
            distance = self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num], obstacle[0], obstacle[1])
            if distance <= (self.pursuer_radius + obstacle[2]):
                return True
        return False

    def cross_border_detect(self,num):
        """跨界检测"""
        if (self.pursuer_pos_x[num] < 0) or (self.pursuer_pos_x[num] > self.width) \
                or (self.pursuer_pos_y[num] < 0) or (self.pursuer_pos_y[num] > self.height)\
                    or (self.evader_pos_x < 0) or (self.evader_pos_x > self.width)\
                        or (self.evader_pos_y < 0) or (self.evader_pos_y > self.height):
            return True
        else:
            return False

    def rep_field(self, scan_info,num):
        """人工势场"""
        distance_list = (1 / (scan_info + 1e-6)) - (1 / self.radius_thr - self.pursuer_radius)
        sin_angel_list = [np.sin((i * np.pi) / (args.pursuer_lidar_lines - 1)) for i in
                          range(1, args.pursuer_lidar_lines - 1)]
        cum_gravitation = np.sum(distance_list * np.array(sin_angel_list))

        return cum_gravitation

    def calculate_angle(self,A, B, C, D):
        """计算ab，cd两条线的夹角，返回0-180角度"""
        # 计算向量 AB 和 CD 的坐标差
        AB = (B[0] - A[0], B[1] - A[1])
        CD = (D[0] - C[0], D[1] - C[1])

        # # 计算向量的长度
        # length_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
        # length_CD = math.sqrt(CD[0] ** 2 + CD[1] ** 2)

        # 计算向量 AB 和 CD 的点积
        dot_product = AB[0] * CD[0] + AB[1] * CD[1]

        # 计算向量 AB 和 CD 的叉积
        cross_product = AB[0] * CD[1] - AB[1] * CD[0]

        # 计算夹角的弧度
        angle_radians = math.atan2(cross_product, dot_product)

        # 将弧度转换为角度
        angle_degrees = math.degrees(angle_radians)

        # 确保角度在 0 到 180 度之间
        if angle_degrees < 0:
            angle_degrees += 180

        return angle_degrees


    def reward_mechanism(self, theta, num):
            """
            以虚拟领导者为中心，同时避开障碍物，1号在leader之前，2,3号在leader之后
            虚拟领航需要考虑：距离与目标越来越近直至重合，运动指向目标，考虑与障碍物碰撞,最大速度
            实际agent需考虑： 出界，障碍物碰撞，相互间碰撞，与evader碰撞，与虚拟领航成对形，距离方向指向evader
            theta: agent与evader 连线角度与agent运动方向的夹角
            r_theta: Rewards and punishments brought by angle
            r_terminal: Rewards and punishments at the end of the round
            r_step: Rewards and punishments in step the environment
            r_distance: Rewards and punishments of distance change
            r_obstacles：Reward and punishment for obstacle avoidance
            r_crash:
            r_total = r_theta + r_terminal + r_step + r_distance
            """
            # 终止状态标志
            self.episode_steps += 1
            done = False
            r_terminal = 0
            r_line = 0

            self.current_distance[num] = self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num],
                                                self.evader_pos_x, self.evader_pos_y)


            # r_crash: 控制各agent之间,agent 与 evader之间的距离, 安全距离为self.pursuer_radius
            p_crash = 0
            r_crash = 0
            r_obstacles = 0
            r_formation = 0
            isCollision = self.collision_detect(num)
            if isCollision or (self.pursuer_pos_x[num] < 0) or (self.pursuer_pos_x[num] > self.width) \
                    or (self.pursuer_pos_y[num] < 0) or (self.pursuer_pos_y[num] > self.height):
                r_terminal = -10
                done = True
                self.isSuccessful[num] = False

            # 避障 r_obstacles
            self.last_scan_info[num, :] = self.scan_info[num, :]
            r_obstacles = -self.K_OBSTACLE * np.sum(1 / (self.scan_info[num, :] + 1e-6) - 1 / self.pursuer_effective_range)
            r_obstacles = np.clip(r_obstacles, -0.5, 0.)

            if num != self.purnum-1:

                if self.current_distance[num] < self.capture_distance:
                    r_terminal = 100
                    self.isSuccessful[num] = True

                for i in range(self.purnum-1):
                  if i != num:
                    distance_uav = self.distance_2d(self.pursuer_pos_x[i], self.pursuer_pos_y[i],
                                                    self.pursuer_pos_x[num],
                                                    self.pursuer_pos_y[num])

                    if (distance_uav < 2 * args.pursuer_safe_radius) and (distance_uav > args.pursuer_safe_radius): #软碰撞
                        # p_crash += 2*np.exp(-2 * distance_uav)/2*np.power(np.cos(dir_uav - self.pursuer_angle[num]),2)# 当distance=2，角度差为pi时，p_crash 接近于0
                        # p_crash = np.exp(-2 * np.pi * distance_uav / np.abs(dir_uav - self.pursuer_angle[num] + 1e-6))
                        p_crash = np.exp(-2 * distance_uav)
                        r_crash += -self.K_CRASH * p_crash
                        self.iscollision[num] = False
                    else:
                        r_crash += -10 #碰撞大大惩罚
                        self.iscollision[num] = True
                # if current_distance < 0.5 * self.pursuer_radius:
                #     r_crash += -10

            if num == self.purnum - 1:
                if self.current_distance[num] < self.capture_distance:
                    r_terminal = 100
                    self.isSuccessful[num] = True


            if (self.episode_steps//self.purnum)+1 >= args.max_episode_steps:  # 当前步数大于阈值，结束，无奖励
                done = True
                if self.current_distance[num] <= self.capture_distance:
                    self.isSuccessful[num] = True
                else:
                    self.isSuccessful[num] = False



            # r_distance
            # r_distance = self.K_DISTANCE * (self.last_distance[num] - current_distance)
            # self.last_distance[num] = current_distance
            #r_distance只考虑与virleader的距离
            d_current =  self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num],
                                                  self.pursuer_pos_x[-1], self.pursuer_pos_y[-1])
            r_distance = self.K_DISTANCE * (self.d_last[num] - d_current)
            self.d_last[num] = d_current
            # r_theta
            # r_theta = (self.K_THETA / np.pi) * ((np.pi / 2) - theta)

            # r_velocity
            # r_velocity = -0.2 * np.clip(args.max_velocity - self.pursuer_velocity[num], 0, 0.5) #让速度接近最大速度
            r_near = abs(self.pursuer_velocity[num]/args.max_velocity) * np.cos(theta)
            # #判断是否已保围目标
            # target_arrest = self.point_in_triangle(np.array([self.pursuer_pos_x[0], self.pursuer_pos_y[0]]), \
            #                                    np.array([self.pursuer_pos_x[1], self.pursuer_pos_y[1]]), \
            #                                    np.array([self.pursuer_pos_x[2], self.pursuer_pos_y[2]]), \
            #                                    np.array(self.evader_pos_x, self.evader_pos_y))
            # formation_step = (np.average(self.current_distance[:3]) < self.capture_distance or target_arrest)
            # if formation_step:
            if num == 0:
                if ((np.abs(self.pursuer_pos_x[num]-self.pursuer_pos_x[-1])< self.capture_distance) and
                        (self.pursuer_pos_y[num]>self.pursuer_pos_y[-1])):
                    r_formation = 100
                else:
                    r_formation = -10
            elif num == 1:
                if (self.pursuer_pos_x[num] < self.evader_pos_x and
                        (abs(self.pursuer_pos_y[num] - self.pursuer_pos_y[num + 1])< 0.5*self.capture_distance) and
                        self.pursuer_pos_y[num]<self.pursuer_pos_y[-1]):
                    r_formation = 100
                else:
                    r_formation = -10
            elif num == 2:
                if (self.pursuer_pos_x[num] > self.evader_pos_x and
                        (abs(self.pursuer_pos_y[num] - self.pursuer_pos_y[num - 1])< 0.5*self.capture_distance) and
                        self.pursuer_pos_y[num] < self.pursuer_pos_y[-1]):
                    r_formation = 100
                else:
                    r_formation = -10


            # r_total
            r_total = r_terminal + r_distance + r_near + r_formation + r_obstacles + r_crash
            return r_total, done


    def angle_normalize(self, angle):
        """角度规则化"""
        # Adjust the angle to (0, 2*np.pi)
        angle %= 2 * np.pi

        return angle

    def velocity_normalize(self, velocity):
        """速度规则化,0到max_velocity 之间"""
        # Adjust the velocity to (0, args.max_velocity)
        velocity = np.clip(velocity, 0, args.max_velocity)

        return velocity

    def reset(self,speed_random = True):
        """环境初始化，返回所有uav状态"""
        # 重置迭代步数
        self.episode_steps = 0
        # 追踪者随机初始化
        self.multi_pursuer_random_reset()
        # 逃避者随机初始化
        self.evader_random_reset()
        # self.evader_reset(speed_random=True) #固定位置，速度，方向
        s=[]
        self.last_distance=[]
        self.d_last = []
        # 距离初始化
        for i in range(self.purnum):
            last_distance = self.distance_2d(self.pursuer_pos_x[i], self.pursuer_pos_y[i],
                                                  self.evader_pos_x, self.evader_pos_y)
            d_last = self.distance_2d(self.pursuer_pos_x[i], self.pursuer_pos_y[i],
                                             self.pursuer_pos_x[-1], self.pursuer_pos_y[-1])
            self.last_distance.append(last_distance)
            self.d_last.append(d_last)
            # 状态初始化
            si = self.pos_to_state(self.pursuer_pos_x[i], self.pursuer_pos_y[i], self.evader_pos_x,
                                  self.evader_pos_y, self.pursuer_angle[i], i)
            s= np.hstack((s,si))
            # s.append(si)
        s = np.array(s).astype(np.float32)

        return s

    def point_in_triangle(self, A, B, C, P):
        # 计算三条边向量
        AB = B - A
        BC = C - B
        CA = A - C

        # 计算向量叉积
        cross_AB_P = np.cross(AB, P - A)
        cross_BC_P = np.cross(BC, P - B)
        cross_CA_P = np.cross(CA, P - C)

        # 判断点是否在三角形内部
        if (cross_AB_P >= 0 and cross_BC_P >= 0 and cross_CA_P >= 0) or \
                (cross_AB_P <= 0 and cross_BC_P <= 0 and cross_CA_P <= 0):
            return True
        else:
            return False
    def step(self, action, isTrain=True):  # action -> [angel, velocity]
        """所有智能体执行动作，返回所有智能体状态, 奖励"""
        taction = []
        for a in action:
        # 将网络输出的动作映射到实际动作
            taction.append(self.get_true_action(a))
        next_state = []
        information = []
        done = []
        reward = []
        theta = []
        self.isSuccessful = np.full((self.purnum), False)
        self.iscollision = np.full((self.purnum), False)
        for num, a in enumerate(taction):

            # 将角度规则化到（0，2*pi）
            self.pursuer_angle[num] = self.angle_normalize(self.pursuer_angle[num] + a[0])
            # 将速度规则化到（0，args.max_velocity）
            self.pursuer_velocity[num] = self.velocity_normalize(self.pursuer_velocity[num] + a[1]) #将实际速度限制在最大速度之内

            pursuer_vel_x = self.pursuer_velocity[num] * np.cos(self.pursuer_angle[num])
            pursuer_vel_y = self.pursuer_velocity[num] * np.sin(self.pursuer_angle[num])

            # pursuer移动，计算位置
            self.pursuer_pos_x[num] = self.pursuer_pos_x[num] + pursuer_vel_x + self.disturbance * np.random.randn()
            self.pursuer_pos_y[num] = self.pursuer_pos_y[num] + pursuer_vel_y + self.disturbance * np.random.randn()

            # 更新中心位置和边界
            self.pursuer[num].set_center((self.pursuer_pos_x[num], self.pursuer_pos_y[num]))
            self.pursuer_field[num].set_center((self.pursuer_pos_x[num], self.pursuer_pos_y[num]))

            # evader移动
        if args.evader_move:
            self.move_evader(gif=not isTrain)
        # distance3 = []
        # 计算速度方向与两线方向的夹角
        for num in range(self.purnum):
            theta.append (self.angel_diff_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num], self.evader_pos_x,
                                   self.evader_pos_y, self.pursuer_angle[num]))
            s_ = self.pos_to_state(self.pursuer_pos_x[num], self.pursuer_pos_y[num], self.evader_pos_x,
                                   self.evader_pos_y, self.pursuer_angle[num], num)
            s_ = np.array(s_).astype(np.float32)
            next_state= np.hstack((next_state, s_))
            # distance4 = self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num], self.evader_pos_x, self.evader_pos_y)
            # distance3.append(distance4)

        # if any(value < self.capture_distance for value in distance3):
        #     cur = 2
        # elif all(value < self.capture_distance for value in distance3):
        #     cur = 3
        # else:
        #     cur = 1
        # 获取奖励信息和终止状态信息
        for num in range(self.purnum):
            r, d= self.reward_mechanism(theta[num], num)
            reward.append(r)
            done.append(d)
            # information.append(info)



        if not isTrain:
            # 判断是否成对形捕获
            point = []
            issuccess = self.point_in_triangle(np.array([self.pursuer_pos_x[0], self.pursuer_pos_y[0]]), \
                                               np.array([self.pursuer_pos_x[1], self.pursuer_pos_y[1]]), \
                                               np.array([self.pursuer_pos_x[2], self.pursuer_pos_y[2]]), \
                                               np.array(self.evader_pos_x, self.evader_pos_y))

            with open('./output_files/pursuer_trajectory.csv', mode='w') as csv_file:  # 根据文件复现目标运行轨迹
                writer = csv.writer(csv_file)
                writer.writerow([self.pursuer_pos_x, self.pursuer_pos_y,self.evader_pos_x,self.evader_pos_y
                                 ])

        # info = '{}/{}'.format(self.episode_steps, args.max_episode_steps)
            return next_state, reward, issuccess, self.iscollision, self.current_distance, self.d_last, self.pursuer_velocity,self.evader_speed

        return next_state, reward, done


    def evader_obs_avoid(self):
        """追踪者避障"""
        is_obs_exist = False
        evader_scan_info = self.evader_lidar.scan_to_state(pos_x=self.evader_pos_x, pos_y=self.evader_pos_y,
                                                           angle=self.evader_angle,
                                                           obstacle_params=self.obstacle_params)  # evader_scan_info: {ndarray: (12,)}
        cum_distance = np.sum(evader_scan_info)
        if np.asarray(evader_scan_info != self.evader_lidar.radius_thr).any():
            is_obs_exist = True

        return is_obs_exist, cum_distance

    def move_evader(self, gif=False):
        """追踪者移动"""
        is_obs_exist, cum_distance = self.evader_obs_avoid()
        # 判断是否改变方向
        if self.evader_pos_x < 20 or self.evader_pos_x > self.width - 20:
            self.evader_speed[0] = -self.evader_speed[0] #离边缘太近，反向运动
        if self.evader_pos_y < 20 or self.evader_pos_y > self.height - 20:
            self.evader_speed[1] = -self.evader_speed[1]
        if is_obs_exist: #如果周围有障碍物，且总距离小于历史距离，反向运动
            if cum_distance < self.cum_distance_history:
                if self.evader_speed[0] > 0:
                    self.evader_speed[0] = np.random.uniform(-args.evader_xy_speed, 0)
                else:
                    self.evader_speed[0] = np.random.uniform(0, args.evader_xy_speed)

                if self.evader_speed[1] > 0:
                    self.evader_speed[1] = np.random.uniform(-args.evader_xy_speed, 0)
                else:
                    self.evader_speed[1] = np.random.uniform(0, args.evader_xy_speed)
        self.cum_distance_history = cum_distance
        # 直线运动 更新位置
        self.evader_pos_x = self.evader_pos_x + self.evader_speed[0]
        self.evader_pos_y = self.evader_pos_y + self.evader_speed[1]
        self.evader_angle = np.arctan2(self.evader_speed[1], self.evader_speed[0])
        self.evader_angle = self.angle_normalize(self.evader_angle)
        # 更新位置
        self.evader.set_center((self.evader_pos_x, self.evader_pos_y))

        if gif:
            with open('./output_files/evader_trajectory.csv', mode='w') as csv_file:  # 根据文件复现目标运行轨迹
                writer = csv.writer(csv_file)
                writer.writerow([self.evader_pos_x, self.evader_pos_y])

    def create_fig_ax(self):
        """创造绘图界面"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # plt.axis('equal')
        plt.grid()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks(np.arange(0, self.width + 1, 25))
        self.ax.set_yticks(np.arange(0, self.height + 1, 25))

        plt.ion()  # 开启interactive mode 成功的关键函数
        self.ax.add_patch(self.evader)
        for i in range(self.purnum):
            self.ax.add_patch(self.pursuer[i])
            self.ax.add_patch(self.pursuer_field[i])
        for obstacle in self.obstacles:
            self.ax.add_patch(obstacle)

        plt.title('X-Y Plot')
        plt.xlabel('X')
        plt.ylabel('Y')

    def render(self,mode='human'):
        """绘图"""

        if mode == 'rgb_array':
            # self.ax.add_patch(self.evader_traj)
            plt.scatter(self.pursuer_pos_x,self.pursuer_pos_y,s = self.pursuer_radius/3,c ='g')
            plt.scatter(self.evader_pos_x, self.evader_pos_y, s = self.pursuer_radius / 3, c='r')
            plt.savefig('./output_images/image_sets/temp' + str(self.episode_steps) + '.png')
            image = imageio.imread('./output_images/image_sets/temp' + str(self.episode_steps) + '.png')
            return image

        elif mode == 'human':
            colors = ['green', 'blue', 'purple', 'yellow','black']
            colors = colors[:self.purnum]
            #
            plt.scatter(self.evader_pos_x, self.evader_pos_y, s=0.5, c='r')
            for i, color in enumerate(colors):
                plt.scatter(self.pursuer_pos_x[i], self.pursuer_pos_y[i], s=0.5, c=color)
        plt.show()
        plt.pause(0.00001)

    def generate_random_color(self):
        # 生成随机的红、绿、蓝三种颜色的值
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # 将颜色值格式化为十六进制字符串，并返回
        return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)

    def move_gif(self):
        """获取动态图"""

        def init():
            self.ax.add_patch(self.evader)
            for i in range(self.purnum):
                self.ax.add_patch(self.pursuer[i])
                self.ax.add_patch(self.pursuer_field[i])
            for obstacle in self.obstacles:
                self.ax.add_patch(obstacle)
            return self.evader, self.pursuer

        def update(frame):
            evader = self.move_evader(gif=True)
            return evader

        anim = animation.FuncAnimation(self.fig, update, init_func=init,
                                       interval=500,
                                       blit=True)
        anim.save('./output_images/movie.gif', writer='pillow')

        plt.close()

        return


    def nearest_target(self,target_list):
        # 为每个pursuer分配target


        nearest_target_list = []
        min_distance_list = []
        min_distance_obs_list = []
        pursuer_angle_list = []
        for i in range(self.purnum):
            min_distance = float('inf')
            min_distance_obs = float('inf')
            nearest_target = None
            for pos in target_list:
                distance = self.distance_2d(self.pursuer_pos_x[i], self.pursuer_pos_y[i], pos[0], pos[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = pos
                for obstacle in self.obstacle_params:
                    distance = self.distance_2d(obstacle[0],obstacle[1],pos[0],pos[1])
                    if distance < min_distance_obs:
                        min_distance_obs = distance
            if len(target_list) > 1:
                target_list.remove(nearest_target)
            nearest_target_list.append(nearest_target)
            min_distance_list.append(min_distance)
            min_distance_obs_list.append(min_distance_obs)
        return nearest_target_list, min_distance_list, min_distance_obs_list

    def check_target(self,target,num):
        """
        判断target中的点是否在uav（x,y）周围障碍物内或者障碍物后面
        在障碍物内返回否，不在障碍物内返回是
        """
        #找出uav能探测到的障碍物
        fov_obs = []  # Obstacle information in UAV field

        for obstacle in self.obstacle_params:
            distance = self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num], obstacle[0], obstacle[1]) - obstacle[2]
            if distance < self.radius_thr:  # 选出在激光束范围内的障碍物
                fov_obs.append(obstacle)

        # 如果周围存在障碍物，判断target是否在障碍物内或者障碍物后面
        if fov_obs:
            for obs in fov_obs:
                circle_center_distance = self.distance_2d(self.pursuer_pos_x[num], self.pursuer_pos_y[num], obs[0], obs[1])
                alpha_ = np.arctan2(obs[1] - self.pursuer_pos_y[num], obs[0] - self.pursuer_pos_x[num])  # uav到障碍物的向量角
                alpha = self.angle_normalize(alpha_)  # 范围0-2pi
                beta_ = np.arcsin((obs[2] + 1e-8) / (circle_center_distance + 1e-8))  # uav到障碍物的向量与uav到障碍物切线向量的夹角,未考虑uav进入障碍物的情况
                beta = self.angle_normalize(beta_)
                distance1 = self.distance_2d(target[0],target[1],self.pursuer_pos_x[num],self.pursuer_pos_y[num])
                distance2 = self.distance_2d(obs[0],obs[1],self.pursuer_pos_x[num],self.pursuer_pos_y[num]) - obs[2]
                angle_low = alpha - beta
                angle_high = alpha + beta
                angle_low = self.angle_normalize(angle_low)
                angle_high = self.angle_normalize(angle_high)  # 确定uav到障碍物边缘的角度范围
                angle = np.arctan2(target[0]-self.pursuer_pos_x[num], target[1]-self.pursuer_pos_y[num])
                angle = self.angle_normalize(angle)  # 范围0-2pi
                if alpha >= beta and (alpha + beta) < 2 * np.pi:
                    return distance1 < distance2 or not (angle_low <= angle <= angle_high)
                else:
                    return distance1 < distance2 or not (((angle >= angle_low) and (angle <= 2 * np.pi))
                                                         or ((angle >= 0) and (angle <= angle_high)))
        # 若周围无障碍物，target为有效值，返回True
        else:
            return True
    def cost(self, target_list):
        # 求所有uav的cost
        w_theta = 0.4
        w_d1 = 0.3
        w_d2 = 0.3
        cost = 0
        pursuer_angle_list = []
        nearest_target_list, min_distance_list, mindistance_obs_list = self.nearest_target(target_list)
        for num, (nearest_target, min_distance, min_distance_obs) in enumerate(zip(nearest_target_list, min_distance_list,mindistance_obs_list)):
            angle_diff = self.angel_diff_2d(self.pursuer_pos_x[num],self.pursuer_pos_y[num],nearest_target[0],nearest_target[1],
                                            self.pursuer_angle[num])
            cost += w_theta * angle_diff + w_d1 * min_distance + w_d2 * max(2-min_distance_obs,0)
            pursuer_angle_list.append(self.angel_diff_2d(self.pursuer_pos_x[num],self.pursuer_pos_y[num],nearest_target[0],nearest_target[1],0))

        return cost,pursuer_angle_list


    def target_point(self):
        #求出中心点，列出所有可能target，排除不可能target，返回cost最小的vertices

        vertices_origin = []
        for i in range(self.purnum):
            vertices_origin.append([self.pursuer_pos_x[i],self.pursuer_pos_y[i]])
        For = Formation()
        # For.calculate_center(vertices_origin)
        For.center = [self.evader_pos_x, self.evader_pos_y] #evader作为队形中心
        min_cost = float('inf')
        final_target_list = None
        pursuer_angle_list = None
        for theta in list(range(0, 360, 10)):
            vertices = For.rotate(theta)
            nearest_target_list, min_distance_list, mindistance_obs_list = self.nearest_target(vertices)
            is_available = []
            for num,target in enumerate(nearest_target_list):
                is_available.append(self.check_target(target,num))
            if all(is_available):
                cost, pursuer_angle = self.cost(nearest_target_list)
                if cost < min_cost:
                    final_target_list = nearest_target_list
                    pursuer_angle_list = pursuer_angle
        return final_target_list, pursuer_angle_list




if (__name__ == '__main__'):
    env = Environment_2D(purnum=4)
    # check_env(env)
    obs = env.reset()
    for i in range(1001):
        action=[]
        a = [-0.5, 0.5]
        for i in range(env.purnum):
            action.append(a)
        isdone = 0
        obs, reward, done, info = env.step(action)
        # final_target_list, pursuer_angle_list = env.target_point()
        # for i in range(len(action)):
        #     action[i][0] = pursuer_angle_list[i]
        # print(obs)
        # print(reward)
        # print(done)
        env.render()
        env.move_gif()
        if all(done):
            obs = env.reset()
            break
