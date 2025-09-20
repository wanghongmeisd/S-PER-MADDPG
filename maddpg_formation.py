import copy
import os
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import csv
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from jueru.utils import get_linear_fn
from jueru.utils import get_latest_run_id
from jueru.Agent_set import DDPG_agent
from jueru.utils import get_obs_shape, get_action_dim

class H_MADDPG:
    """

    """
    def __init__(
            self,
            agent_class_list,
            data_collection_dict_list,
            env: Any,
            updator_dict=None,
            functor_dict_list=None,
            optimizer_dict_list=None,
            lr_dict_list=None,
            exploration_rate: float = 0.1,
            exploration_start: float = 1,
            exploration_end: float = 0.05,
            exploration_fraction: float = 0.2,
            polyak: float = 0.9,
            agent_args_list: List[Dict[str, Any]] = None,
            device: Union[torch.device, str] = "auto",
            max_episode_steps=None,
            eval_func=None,
            gamma: float = 0.95,
            batch_size: int = 512,
            tensorboard_log: str = "./MultiAgent_tensorboard_h/",
            tensorboard_log_name: str = "run",
            render: bool = False,
            action_noise: float = 0.1,
            min_update_step: int = 1000,
            update_step: int = 100,
            start_steps: int = 5000,
            model_address: str = "./MultiAgent_model_address_h",
            save_mode: str = 'step',
            save_interval: int = 5000,
            eval_freq: int = 100,
            eval_num_episode: int = 10,
    ):
        self.agents0 = []
        self.agents1 = []
        self.env = env
        os.makedirs(tensorboard_log, exist_ok=True)
        os.makedirs(model_address, exist_ok=True)
        latest_run_id = get_latest_run_id(tensorboard_log, tensorboard_log_name)
        save_path = os.path.join(tensorboard_log, f"{tensorboard_log_name}_{latest_run_id + 1}")
        os.makedirs(save_path, exist_ok=True)
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.eval_num_episode = eval_num_episode
        self.save_mode = save_mode
        self.eval_freq = eval_freq
        self.model_address = model_address
        self.save_interval = save_interval
        self.writer = SummaryWriter(save_path)
        self.exploration_rate = exploration_rate
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.exploration_fraction = exploration_fraction
        self.exploration_func = get_linear_fn(start=exploration_start,
                                              end=exploration_end,
                                              end_fraction=self.exploration_fraction)

        for i, agent_class in enumerate(agent_class_list):
            if agent_args_list:
                agent_args = agent_args_list[i]
            else:
                agent_args = None

            agent = agent_class(
                functor_dict=functor_dict_list[0][i],
                optimizer_dict=optimizer_dict_list,
                lr_dict=lr_dict_list[i],
                device=self.device,
                **(agent_args or {})
            )

            self.agents0.append(agent)

        for i, agent_class in enumerate(agent_class_list):
            if agent_args_list:
                agent_args = agent_args_list[i]
            else:
                agent_args = None

            agent = agent_class(
                functor_dict=functor_dict_list[1][i],
                optimizer_dict=optimizer_dict_list,
                lr_dict=lr_dict_list[i],
                device=self.device,
                **(agent_args or {})
            )

            self.agents1.append(agent)

        self.updator_dict = updator_dict

        self.data_collection_dict_list = data_collection_dict_list

        self.render = render

        self.action_noise = action_noise

        self.min_update_step = min_update_step

        self.update_step = update_step

        self.batch_size = batch_size

        self.gamma = gamma

        self.polyak = polyak

        self.start_steps = start_steps


    def learn_curriculum_formation(self, num_train_step):
        # 为agent分配网络，两个阶段使用不同网络,self.agent0,self.agent1


        for agent in self.agents0:
            agent.functor_dict['actor'].train()  # model改为训练模式
            agent.functor_dict['critic'].train()
        for agent in self.agents1:
            agent.functor_dict['actor'].train()  # model改为训练模式
            agent.functor_dict['critic'].train()
        # 加载训练好的第一阶段网络
        filepath = 'MultiAgent_model_address_h_local_curriculum_virleader_v1'
        for i, agent in  enumerate(self.agents0):
            address = os.path.join(filepath,f'agent{i}')
            agent = DDPG_agent.load(address)
        for i, agent in  enumerate(self.agents1):
            address = os.path.join(filepath, f'agent{i}')
            agent = DDPG_agent.load(address)

        pursuer_num = self.env.purnum
        obs_space = get_obs_shape(self.env.observation_space)
        virleader_obs_space = get_obs_shape(self.env.virleader_observation_space)
        self.obs_len = obs_space[0]
        self.virleader_obs_len = virleader_obs_space[0]
        negative_infinity = -float('inf')
        episode_num = 0

        average_reward_buf = [negative_infinity] * pursuer_num
        step = 0
        episode_num = 0
        episode_reward = [0] * pursuer_num
        last_episode_reward = [0] * pursuer_num

        while step <= num_train_step:
            for i in range(pursuer_num - 1):
                if last_episode_reward[i] < episode_reward[i]:
                    last_episode_reward[i] = episode_reward[i]
            # last_episode_reward = episode_reward.copy()
            episode_reward = [0] * pursuer_num
            # episode_step = [0] * pursuer_num
            state = self.env.reset()
            info = [False,False,False,False]
            while True:
                action = []

                if self.render:
                    self.env.render()
                if all(info[:3]):
                    agents = self.agents1
                else:
                    agents = self.agents0
                for i, agent in enumerate(agents):
                    state_cuda = torch.tensor(state).to(self.device)
                    if i != pursuer_num - 1:
                        action.append(agent.choose_action(state_cuda[self.obs_len * i:self.obs_len * (i + 1)],
                                                          self.action_noise))  # 领航者动作也加上noise，进行训练
                    else:
                        action.append(agent.choose_action(state_cuda[-self.virleader_obs_len:], self.action_noise))

                next_state, reward, done,info = self.env.step(action) #先训练navigation
                done_value = [0.0 if value == 'True' else 1.0 for value in done]
                # whole_state = np.array([element for row in state for element in row])
                # next_whole_state = np.array([element for row in next_state for element in row])
                whole_action = np.array([element for row in action for element in row])
                if all(info[:3]):
                    self.data_collection_dict_list[1].store_whole_cuda(whole_obs=state, whole_act=whole_action,
                                                                whole_rew=reward,
                                                                next_whole_obs=next_state,
                                                                whole_done=done_value)  # 存放总的状态和reward
                else:
                    self.data_collection_dict_list[0].store_whole_cuda(whole_obs=state, whole_act=whole_action,
                                                                       whole_rew=reward,
                                                                       next_whole_obs=next_state,
                                                                       whole_done=done_value)  # 存放总的状态和reward
                state = next_state.copy()
                episode_reward = [x + y for x, y in zip(episode_reward, reward)]
                step += 1

                if step >= self.min_update_step and step % self.update_step == 0:
                    new_row=[]
                    #训练第一阶段网络
                    for i, agent in enumerate(self.agents0):
                        for j in range(self.update_step):
                            batch = self.data_collection_dict_list[0].sample_batch_whole(self.batch_size)
                            if i != pursuer_num - 1:  # 用全局状态，所有动作训练
                                critic_loss = self.updator_dict['critic_update'](env=self.env,
                                                                                 agents=self.agents0,
                                                                                 num=i,
                                                                                 whole_state=batch['state'],
                                                                                 whole_action=batch['action'],
                                                                                 whole_reward=batch['reward'],
                                                                                 next_whole_state=batch['next_state'],
                                                                                 whole_done_value=batch['done'],
                                                                                 gamma=self.gamma)
                            if i == pursuer_num - 1:
                                critic_loss = self.updator_dict['virleader_critic_update'](env=self.env,
                                                                                           agent=agent,
                                                                                           state=batch['state'],
                                                                                           action=batch['action'],
                                                                                           reward=batch['reward'],
                                                                                           next_state=batch[
                                                                                               'next_state'],
                                                                                           done_value=batch['done'],
                                                                                           gamma=self.gamma)
                            if j % 4 == 0:
                                if i != pursuer_num - 1:
                                    actor_loss = self.updator_dict['actor_update'](env=self.env,
                                                                                   agents=self.agents0,
                                                                                   num=i,
                                                                                   whole_state=batch['state'],
                                                                                   )
                                if i == pursuer_num - 1:
                                    actor_loss = self.updator_dict['virleader_actor_update'](env=self.env,
                                                                                             agent=agent,
                                                                                             state=batch['state'])

                                self.updator_dict['soft_update'](agent.functor_dict['actor_target'],
                                                                 agent.functor_dict['actor'],
                                                                 polyak=self.polyak)

                                self.updator_dict['soft_update'](agent.functor_dict['critic_target'],
                                                                 agent.functor_dict['critic'],
                                                                 polyak=self.polyak)


                            self.writer.add_scalar(f'{i}_critic_loss_step1', critic_loss,
                                                   global_step=(step + j))
                            self.writer.add_scalar(f'{i}_actor_loss_step1', actor_loss,
                                                   global_step=(step + j))
                    # 训练第二阶段网络
                    for i, agent in enumerate(self.agents1):
                        for j in range(self.update_step):
                            batch = self.data_collection_dict_list[1].sample_batch_whole(self.batch_size)
                            if i != pursuer_num - 1:  # 用全局状态，所有动作训练
                                critic_loss = self.updator_dict['critic_update'](env=self.env,
                                                                                 agents=self.agents1,
                                                                                 num=i,
                                                                                 whole_state=batch['state'],
                                                                                 whole_action=batch['action'],
                                                                                 whole_reward=batch['reward'],
                                                                                 next_whole_state=batch[
                                                                                     'next_state'],
                                                                                 whole_done_value=batch['done'],
                                                                                 gamma=self.gamma)
                            if i == pursuer_num - 1:
                                critic_loss = self.updator_dict['virleader_critic_update'](env=self.env,
                                                                                           agent=agent,
                                                                                           state=batch['state'],
                                                                                           action=batch[
                                                                                               'action'],
                                                                                           reward=batch[
                                                                                               'reward'],
                                                                                           next_state=batch[
                                                                                               'next_state'],
                                                                                           done_value=batch[
                                                                                               'done'],
                                                                                           gamma=self.gamma)
                            if j % 4 == 0:
                                if i != pursuer_num - 1:
                                    actor_loss = self.updator_dict['actor_update'](env=self.env,
                                                                                   agents=self.agents1,
                                                                                   num=i,
                                                                                   whole_state=batch['state'],
                                                                                   )
                                if i == pursuer_num - 1:
                                    actor_loss = self.updator_dict['virleader_actor_update'](env=self.env,
                                                                                             agent=agent,
                                                                                             state=batch[
                                                                                                 'state'])

                                self.updator_dict['soft_update'](agent.functor_dict['actor_target'],
                                                                 agent.functor_dict['actor'],
                                                                 polyak=self.polyak)

                                self.updator_dict['soft_update'](agent.functor_dict['critic_target'],
                                                                 agent.functor_dict['critic'],
                                                                 polyak=self.polyak)

                            self.writer.add_scalar(f'{i}_critic_loss_step2', critic_loss,
                                                   global_step=(step + j))
                            self.writer.add_scalar(f'{i}_actor_loss_step2', actor_loss,
                                                   global_step=(step + j))
                if any(done):
                    episode_num += 1
                    for i, agent in enumerate(agents):
                        self.writer.add_scalar(f'agent_{i}_episode_reward', episode_reward[i], global_step=step)
                    print(f'episode_num: {episode_num}, episode_reward: {episode_reward}, golbal_step: {step}')
                    if self.save_mode == 'eval':
                        if step >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance_multi_agent(num_episode=self.eval_num_episode,
                                                                               step=step)
                            if sum(average_reward) > sum(average_reward_buf):
                                for i, agent in enumerate(self.agents0):
                                    address = os.path.join(self.model_address, f'agent{i}_step1')
                                    agent.save(address)
                                for i, agent in enumerate(self.agents1):
                                    address = os.path.join(self.model_address, f'agent{i}_step2')
                                    agent.save(address)
                                average_reward_buf = average_reward

                        # with open('multiagent_rewards.txt', 'a') as file:
                        #     file.write(f'{episode_reward}\n')
                        # 每隔一段时间存储一次agent参数

                    if self.save_mode == 'step':
                        if episode_reward[i] > 0 and episode_reward[i] > last_episode_reward[i]:
                            address = os.path.join(self.model_address, f'agent{i}')
                            agent.save(address)

                    break
            self.writer.close()
    def eval_performance_multi_agent(self, num_episode, step):
         #同时评估所有agent的reward

        list_episode_rewards = []
        state = self.env.reset()
        success_num = 0
        info = [False,False,False,False]
        for _ in range(num_episode):
            episode_reward = [0] * (self.env.purnum)
            while True:
                triangle = []
                action = []
                if not all(info[:3]):
                    agents = self.agents0
                else:
                    agents=self.agents1
                for i, agent in enumerate(agents):
                    if i != self.env.purnum - 1:
                        state_cuda = torch.tensor(state[self.obs_len * i:self.obs_len * (i + 1)]).to(self.device)
                        action.append(agent.predict(state_cuda))
                        triangle.append((self.env.pursuer_pos_x[i], self.env.pursuer_pos_y[i]))

                    else:
                        state_cuda = torch.tensor(state[-self.virleader_obs_len:]).to(self.device)
                        action.append(agent.predict(state_cuda))
                point = (self.env.evader_pos_x,self.env.evader_pos_y)
                isevader_in_pursuer = self.point_in_triangle(point,triangle)
                if isevader_in_pursuer:
                    success_num += 1
                next_state, reward, done,info = self.env.step(action)
                state = next_state.copy()
                episode_reward = np.add(episode_reward, reward)


                if any(done):
                    # if self.two_greater_than_zero(episode_reward) > 0:
                    #     success_num += 1
                    list_episode_rewards.append(episode_reward)
                    break
        average_reward = np.mean(list_episode_rewards, axis=0) #计算数组列的平均值
        success_rate = success_num / num_episode
        for num in range(self.env.purnum):
            self.writer.add_scalar(f'{num}_eval_average_reward', average_reward[num], global_step=step)
        self.writer.add_scalar(f'eval_success_rate', success_rate, global_step=step)

        return average_reward
    def eval_performance_multi_agent_noleader(self, num_episode, step):
         #同时评估所有agent的reward

        list_episode_rewards = []
        state = self.env.reset_noleader()
        success_num = 0

        for _ in range(num_episode):
            episode_reward = [0] * (self.env.purnum)
            while True:
                triangle = []
                action = []
                for i, agent in enumerate(self.agents):

                    state_cuda = torch.tensor(state[self.obs_len * i:self.obs_len * (i + 1)]).to(self.device)
                    action.append(agent.predict(state_cuda))  # 领航者动作也加上noise，进行训练
                    triangle.append((self.env.pursuer_pos_x[i], self.env.pursuer_pos_y[i]))
                point = (self.env.evader_pos_x,self.env.evader_pos_y)
                isevader_in_pursuer = self.point_in_triangle(point,triangle)
                if isevader_in_pursuer:
                    success_num += 1
                next_state, reward, done = self.env.step_noleader(action)
                state = next_state.copy()
                episode_reward = np.add(episode_reward, reward)

                if all(done):
                    # if self.two_greater_than_zero(episode_reward) > 0:
                    #     success_num += 1
                    list_episode_rewards.append(episode_reward)
                    break
        average_reward = np.mean(list_episode_rewards, axis=0) #计算数组列的平均值
        success_rate = success_num / num_episode
        for num in range(self.env.purnum):
            self.writer.add_scalar(f'{num}_eval_average_reward', average_reward[num], global_step=step)
        self.writer.add_scalar(f'eval_success_rate', success_rate, global_step=step)

        return average_reward

    def two_greater_than_zero(self, lst):
        count = 0
        for num in lst:
            if num > 0:
                count += 1
                if count >= 2:
                    return True
        return False

    def point_in_triangle(self,point,triangle):
        """
        判断点是否在三角形内

        参数：
        point: 一个元组，表示点的坐标，例如 (x, y)
        triangle: 一个包含三个元组的列表，每个元组表示三角形的一个顶点，例如 [(x1, y1), (x2, y2), (x3, y3)]

        返回值：
        True 如果点在三角形内，否则 False
        """

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(point, triangle[0], triangle[1]) < 0.0
        b2 = sign(point, triangle[1], triangle[2]) < 0.0
        b3 = sign(point, triangle[2], triangle[0]) < 0.0

        return (b1 == b2) and (b2 == b3)


