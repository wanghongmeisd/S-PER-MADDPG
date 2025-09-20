"""与train_ddpg_virleader_1 改变了 环境，critic 512*256"""
import numpy as np
from jueru.Agent_set import Agent, DDPG_agent
# from jueru.algorithms import BaseAlgorithm, MAAlgorithm, MAGAAlgorithms
from maddpg import H_MADDPG
from jueru.datacollection_multi import Replay_buffer, SingleReplay_buffer, GAIL_DataSet
from jueru.updator_multi import critic_updator_ddpg_localstate_virleader,actor_updator_ddpg_localstate_virleader,soft_update,\
                                    virleader_actor_updator_ddpg,virleader_critic_updator_ddpg
from jueru.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic,FlattenExtractor
from multi_uav_env_virleader_4 import Environment_2D
from jueru.utils import get_obs_shape, get_action_dim
import time
import torch
import csv
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#cpu 起始时间
start_time = time.time()
# 创建 CUDA 事件对象
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
# 记录cuda起始时间
start_event.record()
purnum = 4
env = Environment_2D(purnum=purnum)
# env.reset_virleader
DDPG_agent_class_list = []
# data_collection_list = []
discriminator_list = []
functor_dict_list=[]
lr_dict_list=[]
feature_dim = 128
# whole_feature_extractor = []
# whole_action_space = []
# obs_shape = get_obs_shape(env.observation_space)
# action_dim = get_action_dim(env.action_space)
feature_extractor = FlattenExtractor(env.observation_space)# agent的状态
feature_extractor_virleader = FlattenExtractor(env.virleader_observation_space)# 将状态转化成一维张量
# whole_feature_extractor = FlattenExtractor(env.whole_observation_space)#所有agent状态
seed = 10
actor = ddpg_actor(env.action_space, feature_extractor,[256,256],seed= seed).to(device)
actor_virleader = ddpg_actor(env.action_space, feature_extractor_virleader,[128,128],seed= seed).to(device)
critic = ddpg_critic(env.action_space, feature_extractor,[512,256],seed= seed).to(device) # 输入包括virleader在内的四个agent的状态
critic_virleader = ddpg_critic(env.action_space, feature_extractor_virleader,[128,128],seed= seed).to(device)
functor_dict = {}
lr_dict = {}
updator_dict = {}

functor_dict_list = [{'actor':actor, 'critic': critic,'actor_target':actor,'critic_target':critic},
                     {'actor':actor, 'critic': critic,'actor_target':actor,'critic_target':critic},
                     {'actor':actor, 'critic': critic,'actor_target':actor,'critic_target':critic},
                     {'actor':actor_virleader, 'critic': critic_virleader,'actor_target':actor_virleader,'critic_target':critic_virleader}]

lr_dict['actor'] = 1e-4
lr_dict['critic'] = 1e-4
lr_dict['actor_target'] = 1e-4
lr_dict['critic_target'] = 1e-4


updator_dict['actor_update'] = actor_updator_ddpg_localstate_virleader
updator_dict['critic_update'] = critic_updator_ddpg_localstate_virleader
updator_dict['soft_update'] = soft_update
updator_dict['virleader_actor_update'] = virleader_actor_updator_ddpg
updator_dict['virleader_critic_update'] = virleader_critic_updator_ddpg
# 一个replay——buffer，存放所有uav状态
data_collection_list = Replay_buffer(env=env, size=2e6, device=device) #存放在 cuda
for agent_name in range(env.purnum):
    DDPG_agent_class_list.append(DDPG_agent)
    discriminator_list.append(None)
    lr_dict_list.append(lr_dict)
    # functor_dict_list.append({'actor':actor, 'critic': critic,'actor_target':actor,'critic_target':critic})
    # data_collection_list.append(Replay_buffer(env=env, size=2e6, device="cpu"))# 最后一个存放全局状态

# Assuming you have defined your agent_class_list, data_collection_dict_list, env_list, etc.

# Instantiate MultiAgentAlgorithm

MADDPG = H_MADDPG(
    agent_class_list = DDPG_agent_class_list,
    data_collection_dict_list = data_collection_list,
    env = env,
    updator_dict=updator_dict,
    functor_dict_list=functor_dict_list,
    lr_dict_list=lr_dict_list,
    exploration_rate=0.1,
    exploration_start=1,
    exploration_end=0.05,
    exploration_fraction=0.2,
    polyak=0.9,
    device=device,  # or "cpu" or "auto" based on your preference
    max_episode_steps=1000,
    gamma=0.95,
    batch_size=200,
    tensorboard_log="./MultiAgent_tensorboard_h_virleader_v4_seed=10/",
    tensorboard_log_name="run",
    render=False,
    action_noise=0.1,
    min_update_step=10000,
    update_step=100,
    start_steps=2000,
    model_address="MultiAgent_model_address_h_virleader_v4_seed=10",
    save_mode='eval',
    save_interval=2000,
    eval_freq=20,
    eval_num_episode=5,

)

# Train the agents
num_train_steps =10000000 # You can set the desired number of training steps
MADDPG.learn_curriculum(num_train_steps)
end_time = time.time()
# 记录结束时间
end_event.record()
# 等待 CUDA 操作完成
torch.cuda.synchronize()
# 计算时间间隔
cuda_time_ms = start_event.elapsed_time(end_event)
cuda_time_sec = cuda_time_ms/1000
cpu_time_sec = end_time - start_time
print("CUDA 运行时间:", cuda_time_sec, "秒")
print("cpu运行时间为: {:.2f} 秒".format(cpu_time_sec))






