# -*- coding:utf-8 -*-
"""
作者：509
日期：2021年01月31日
"""
import numpy as np

class Lidar:
    def __init__(self, lidar_lines=8, isTotal_range=True, radius_thr=10):
        """
        Function description: Simulation of lidar, scanning the surrounding environment to obtain range information.
        :param lidar_lines: Number of rays emitted by lidar.
        :param isTotal_range: Scanning mode of lidar.
        :param radius_thr: Maximum scanning range of lidar.
        """
        self.lidar_lines = lidar_lines
        self.isTotal_range = isTotal_range
        self.radius_thr = radius_thr
        self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)# lidar_lines个 值为radius_thr的一维数组

    def get_env_state(self, pos_x, pos_y, angle, obstacle_params):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.angle = angle
        self.obstacle_params = obstacle_params #障碍物位置和大小信息

    def distance_2d(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    def angle_normalize(self, angle):
        # Adjust the angle to (0, 2*np.pi)
        angle %= 2 * np.pi

        return angle

    def scan_distance(self, circle_center_distance, angle, obs):
        '''计算 穿过障碍物内的激光束到圆形障碍物边缘的  距离'''
        line_para = np.array([np.tan(angle), -1, self.pos_y - (self.pos_x * np.tan(angle))])#角度为angle激光束的函数
        dot_para = np.array([obs[0], obs[1], 1])
        dot_line_distance = np.abs(line_para @ dot_para) / np.sqrt(np.square(line_para[0]) + 1) # @矩阵乘法，障碍物中心到激光束的距离
        distance_long = np.sqrt(np.square(circle_center_distance) - np.square(dot_line_distance)) #最长距离，激光束上uav到在障碍物内弦垂点的距离
        distance_short = np.sqrt(np.square(obs[2]) - np.square(dot_line_distance))#最短距离，弦的一半
        distance_true = distance_long - distance_short #激光束到圆形障碍物边缘的距离

        return distance_true

    def distance_min(self, distance_trues):
        for index, distance_true in distance_trues:
            self.scan_info[index] = np.min([self.scan_info[index], distance_true])

    def get_scan_info(self, fov_obs, angles):
        # Get scan information，只适用于圆形障碍物。若是长方形障碍物可以分成三个区域计算真实距离
        line_angles = []
        distance_trues = []
        for obs in fov_obs:
            circle_center_distance = self.distance_2d(self.pos_x, self.pos_y, obs[0], obs[1])
            alpha_ = np.arctan2(obs[1] - self.pos_y, obs[0] - self.pos_x) #uav到障碍物的向量角
            alpha = self.angle_normalize(alpha_)#范围0-2pi

            beta_ = np.arcsin((obs[2]+1e-8) / (circle_center_distance + 1e-8)) #uav到障碍物的向量与uav到障碍物切线向量的夹角,未考虑uav进入障碍物的情况
            beta = self.angle_normalize(beta_)
            if alpha >= beta and (alpha + beta) < 2 * np.pi:
                angle_low = alpha - beta
                angle_high = alpha + beta
                angle_low = self.angle_normalize(angle_low)
                angle_high = self.angle_normalize(angle_high)#确定uav到障碍物边缘的角度范围
                for index, angle in enumerate(angles):# 判断在这个范围内的激光束，确定激光束返回值
                    if (angle >= angle_low) and (angle <= angle_high):
                        line_angles.append((index, angle))
                        distance_true = self.scan_distance(circle_center_distance, angle, obs)
                        distance_trues.append((index, distance_true))
                self.distance_min(distance_trues)
            else:
                angle_low = alpha - beta
                angle_high = alpha + beta
                angle_low = self.angle_normalize(angle_low)
                angle_high = self.angle_normalize(angle_high)
                for index, angle in enumerate(angles):
                    if ((angle >= angle_low) and (angle <= 2 * np.pi)) or ((angle >= 0) and (angle <= angle_high)):
                        line_angles.append((index, angle))
                        distance_true = self.scan_distance(circle_center_distance, angle, obs)
                        distance_trues.append((index, distance_true))
                self.distance_min(distance_trues)

    def scan_environment(self):
        # Giving UAVs the ability to scan the environment, 返回距离和激光束的角度
        fov_obs = []  # Obstacle information in UAV field
        fov_bos_radius = []
        angles = []
        for obstacle in self.obstacle_params:

            distance = self.distance_2d(self.pos_x, self.pos_y, obstacle[0], obstacle[1]) - obstacle[2]
            if distance < self.radius_thr: #选出在激光束范围内的障碍物
                fov_obs.append(obstacle)
                fov_bos_radius.append(obstacle[2])
        if len(fov_obs) != 0:
            if self.isTotal_range:  # Global range scanning
                if self.lidar_lines % 2:
                    self.lidar_lines += 1
                # scan_info: Information obtained by lidar scanning environment
                self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)
                single_angle = (2 * np.pi) / self.lidar_lines
                for i in range(0, self.lidar_lines):
                    angle_add = self.angle + (i * single_angle)
                    angle = self.angle_normalize(angle_add)
                    angles.append(angle)
                self.get_scan_info(fov_obs, angles)
            else:  # Semi global range scanning
                if self.lidar_lines % 2 == 0:
                    self.lidar_lines += 1
                # scan_info: Information obtained by lidar scanning environment
                self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)
                single_angle = np.pi / (self.lidar_lines - 1)
                for i in range(-(self.lidar_lines // 2), self.lidar_lines // 2 + 1):
                    angle_add = self.angle + (i * single_angle)
                    angle = self.angle_normalize(angle_add)
                    angles.append(angle)
                self.get_scan_info(fov_obs, angles)

    def scan_to_state(self, pos_x, pos_y, angle, obstacle_params):  # Interface API
        self.get_env_state(pos_x, pos_y, angle, obstacle_params)
        self.scan_environment()

        return self.scan_info