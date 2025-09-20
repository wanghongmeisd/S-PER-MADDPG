# from multi_uav_env_virleader_fouragent_4 import Environment_2D
from multi_uav_env_virleader_5 import Environment_2D
from jueru.Agent_set import DDPG_agent
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from jueru.utils import get_latest_run_id
import os
import torch
from jueru.utils import get_obs_shape, get_action_dim
import numpy as np


if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tensorboard_log = "./test_files/"
    tensorboard_log_name = "run"
    env = Environment_2D(disturbance=0,purnum=4)
    # filepath = 'MultiAgent_model_address_h_virleader_maddpg_4'
    # filepath = 'MultiAgent_model_address_h_virleader_v4_seed=10'
    # filepath = 'MultiAgent_model_address_h_virleader_fouragent'
    # filepath = 'MultiAgent_model_address_h_virleader_fouragent'
    # filepath = 'MultiAgent_model_address_h_local_per_4 seed=10'
    # filepath='MultiAgent_model_address_h_virleader_v3'
    # filepath='MultiAgent_model_address_h_virleader_maddpg_4_seed=10'
    filepath='MultiAgent_model_address_h_virleader_maddpg_5 seed=10'
    # filepath='MultiAgent_model_address_per_virleader_fouragent_4_seed=10'
    agentlist=[]
    for i in range(env.purnum):
        path = os.path.join(filepath,f'agent{i}')#调用分层训练的maddpg
        agentlist.append(DDPG_agent.load(path))

    latest_run_id = get_latest_run_id(tensorboard_log, tensorboard_log_name)
    save_path = os.path.join(tensorboard_log, f"{tensorboard_log_name}_{latest_run_id + 1}")
    writer = SummaryWriter(save_path)
    # env.render(mode='human')
    target_count = 0
    step = 0
    test_num = 50
    ndone = 0
    obs_space = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    virleader_obs_len = virleader_obs_space[0]
    # env.record_video()


    for i in range(test_num):
        state = env.reset(speed_random=False)
        collision_num = 0
        success_num = 0
        capture_num = 0
        for j in range(200):
            state_cuda = torch.tensor(state).to(device)
            # ndone = 0
            env.render(mode='human')
            # env.move_gif()
            action=[]
            for k, agent in enumerate(agentlist):
                step += 1
                if k != env.purnum-1:
                    action.append(agent.predict(state_cuda[obs_len * k:obs_len * (k + 1)]))
                else:
                    action.append(agent.predict(state_cuda[-virleader_obs_len:]))

            # print(action)
            next_state, reward, issuccess, iscapture, iscollision, current_distance, d_last, pursuer_velocity,evader_speed= env.step(action,isTrain=False)
            is_success = 1 if issuccess else 0
            iscollision = iscollision.astype(int)
            if all(not x for x in iscapture[:env.purnum-1]):
                collision_num += sum(iscollision[:env.purnum-1])
            else:
                capture_num += 1
            success_num += is_success
            evader_velocity = np.sqrt(np.square(evader_speed[0]) + np.square(evader_speed[1]))
            # collision_rate = collision_num/300
            # success_rate = success_num/300
            state = next_state.copy()
            for num in range(env.purnum-1):
                writer.add_scalar(f'agent{num} to virleader',d_last[num],j)
                writer.add_scalar(f'agent{num} velocity', pursuer_velocity[num], j)
                writer.add_scalar(f'agent{num} to evader', current_distance[num],j)
            writer.add_scalar(f'evader_velocity', evader_velocity, j)
        writer.add_scalar(f'success_num', success_num, i)
        writer.add_scalar(f'collision_num', collision_num, i)
        writer.add_scalar(f'capture_num', capture_num, i)
                # env.record_video()
            # if all(done):
            #     # plt.clf()
            #     obs = env.reset(speed_random=True)
            #     ndone += 1
            #     break
    # eval(env, agent, 1000)


    env.close()