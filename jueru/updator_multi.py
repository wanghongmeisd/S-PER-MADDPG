import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jueru.utils import get_obs_shape, get_action_dim


def soft_update(target, source, polyak):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              states, actions, rewards, next_states, masks, gamma, polyak):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    """update critic"""

    values = value_net(states, actions)

    with torch.no_grad():
        target_next_values = value_net_target(next_states, policy_net_target(next_states))
        target_values = rewards + gamma * masks * target_next_values
    value_loss = nn.MSELoss()(values, target_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update actor"""

    policy_loss = - value_net(states, policy_net(states)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """soft update target nets"""
    soft_update(policy_net_target, policy_net, polyak)
    soft_update(value_net_target, value_net, polyak)
    return value_loss, policy_loss


def critic_updator_ddpg(agent, state, action, reward, next_state, done_value, gamma, ):
    value = agent.functor_dict['critic'](state, action)
    # print(value.mean())
    with torch.no_grad():
        target_next_value = agent.functor_dict['critic_target'](next_state, agent.functor_dict['actor_target'](next_state))
        #print('reward', reward)
        #print(done_value)
        target_value = reward + gamma * (done_value) * target_next_value

    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value[0])
    # print('value', value[0])

    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss


def critic_updator_ddpg_per(agent, state, action, reward, next_state, done_value, weight, gamma ):
    """
    使用权重值修改方差公式，更新critic，并返回sample的td-error
    """
    value = agent.functor_dict['critic'](state, action)
    # print(value.mean())
    with torch.no_grad():
        target_next_value = agent.functor_dict['critic_target'](next_state,
                                                                agent.functor_dict['actor_target'](next_state))
        # print('reward', reward)
        # print(done_value)
        target_value = reward + gamma * (done_value) * target_next_value

        # 计算平方差并求平均
    td_error = target_value - value
    value_loss = torch.sum(weight * torch.square(td_error)) / len(value)
    # value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss,td_error
# virleader critic 更新
def virleader_critic_updator_ddpg(env, agent, state, action, reward, next_state, done_value, gamma, ):
    """
          含有virleader系统的virleader 的critic的更新
          """
    obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    action_dim = get_action_dim(env.action_space)
    value = agent.functor_dict['critic'](state[:,-obs_len:], action[:,-action_dim:])
    # print(value.mean())
    with torch.no_grad():
        target_next_value = agent.functor_dict['critic_target'](next_state[:,-obs_len:], agent.functor_dict['actor_target'](next_state[:,-obs_len:]))
        #print('reward', reward)
        #print(done_value)
        virleader_reward = reward[:,-1].reshape(-1,1)
        virleader_done = done_value[:,-1].reshape(-1,1)
        target_value = virleader_reward + gamma * virleader_done * target_next_value
    value_loss = nn.MSELoss()(value, target_value)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss
# virleader 状态与其它agent状态不一样
def critic_updator_maddpg(env,agents,num, whole_state,whole_action,whole_reward, next_whole_state, whole_done_value, gamma, ):
    """
    maddpg,输入所有agent的动作和状态
    """

    obs_space = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    virleader_obs_len = virleader_obs_space[0]
    action_dim = get_action_dim(env.action_space)
    next_whole_action = []
    purnum = env.purnum
    ACTION = whole_action[:, 0:-2]
    value = agents[num].functor_dict['critic'](whole_state[:, :-virleader_obs_len], whole_action[:,0:-2])
    # print(value.mean())
    with torch.no_grad():
        for i, agent in enumerate(agents):
            if i != purnum-1:
                input_state = next_whole_state[:, obs_len * num:obs_len * (num + 1)]
                next_action = agent.functor_dict['actor_target'](input_state) # 200*2
                next_whole_action.append(next_action) # 包含 所有agent一个batch的action ,3个 100*2 tensor
            # else:
            #     input_state = next_whole_state[:, -virleader_obs_len:]
            #     next_action = agent.functor_dict['actor_target'](input_state)  # 100*2
            #     next_whole_action.append(next_action)  # 包含 所有agent一个batch的action ,4个 100*2 tensor
        # 初始化结果张量
        concatenated_tensor = next_whole_action[0]
        # 循环按列拼接张量
        for i in range(1, len(next_whole_action)):
            concatenated_tensor = torch.cat([concatenated_tensor, next_whole_action[i]], dim=1) # 将torch框架下的张量action按列拼接


        target_next_value = agents[num].functor_dict['critic_target'](next_whole_state[:, :-virleader_obs_len], concatenated_tensor)
        #print('reward', reward)
        #print(done_value)
        whole = whole_reward[:,num].reshape(-1,1) # 第num个agent的奖励值
        whole1 = whole_done_value[:,num].reshape(-1,1)#第num个agent的done值
        target_value = whole + gamma * whole1 * target_next_value
        # noise_range = 0.05  # 噪声范围为目标值的5%
        # # 生成随机噪声
        # noise = np.random.uniform(low=-noise_range * target_value, high=noise_range * target_value)
        # # 添加噪声
        # noise = noise.astype(np.float32)
        # target_value += noise

    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agents[num].optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agents[num].optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss
def critic_updator_maddpg_1(env,agents,num, whole_state,whole_action,whole_reward, next_whole_state, whole_done_value, gamma, ):
    """
    maddpg,输入所有agent的动作和位置信息
    """

    obs_space = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    virleader_obs_len = virleader_obs_space[0]
    action_dim = get_action_dim(env.action_space)
    next_whole_action = []
    purnum = env.purnum
    ACTION = whole_action[:, 0:-2]
    value = agents[num].functor_dict['critic'](whole_state[:,obs_len*num:obs_len * (num + 1)], whole_action)
    # value = agents[num].functor_dict['critic'](whole_state, whole_action)
    # print(value.mean())
    with torch.no_grad():
        for i, agent in enumerate(agents):
            if i != purnum-1:
                input_state = next_whole_state[:, obs_len * i:obs_len * (i + 1)]
            else:
                input_state = next_whole_state[:, -virleader_obs_len:]
            next_action = agent.functor_dict['actor_target'](input_state) # 200*2
            next_whole_action.append(next_action) # 包含 所有agent一个batch的action ,3个 100*2 tensor
            # else:
            #     input_state = next_whole_state[:, -virleader_obs_len:]
            #     next_action = agent.functor_dict['actor_target'](input_state)  # 100*2
            #     next_whole_action.append(next_action)  # 包含 所有agent一个batch的action ,4个 100*2 tensor
        # 初始化结果张量
        concatenated_tensor = next_whole_action[0]
        # 循环按列拼接张量
        for i in range(1, len(next_whole_action)):
            concatenated_tensor = torch.cat([concatenated_tensor, next_whole_action[i]], dim=1) # 将torch框架下的张量action按列拼接


        target_next_value = agents[num].functor_dict['critic_target'](next_whole_state[:,obs_len*num:obs_len * (num + 1)], concatenated_tensor)
        #print('reward', reward)
        # target_next_value = agents[num].functor_dict['critic_target'](
        #     next_whole_state, concatenated_tensor)
        #print(done_value)
        whole = whole_reward[:,num].reshape(-1,1) # 第num个agent的奖励值
        whole1 = whole_done_value[:,num].reshape(-1,1)#第num个agent的done值
        target_value = whole + gamma * whole1 * target_next_value

        # noise_range = 0.05  # 噪声范围为目标值的5%
        # # 生成随机噪声
        # noise = np.random.uniform(low=-noise_range * target_value, high=noise_range * target_value)
        # # 添加噪声
        # noise = noise.astype(np.float32)
        # target_value += noise

    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agents[num].optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agents[num].optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss
def critic_updator_maddpg_per_1(env,agents,num, whole_state,whole_action,whole_reward, next_whole_state, whole_done_value, weight,gamma, ):
    """
    maddpg,输入所有agent的动作和位置信息
    """

    obs_space = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    virleader_obs_len = virleader_obs_space[0]
    action_dim = get_action_dim(env.action_space)
    next_whole_action = []
    purnum = env.purnum
    ACTION = whole_action[:, 0:-2]
    value = agents[num].functor_dict['critic'](whole_state[:,obs_len*num:obs_len * (num + 1)], whole_action)
    # value = agents[num].functor_dict['critic'](whole_state, whole_action)
    # print(value.mean())
    with torch.no_grad():
        for i, agent in enumerate(agents):
            if i != purnum-1:
                input_state = next_whole_state[:, obs_len * i:obs_len * (i + 1)]
            else:
                input_state = next_whole_state[:, -virleader_obs_len:]
            next_action = agent.functor_dict['actor_target'](input_state) # 200*2
            next_whole_action.append(next_action) # 包含 所有agent一个batch的action ,4个 100*2 tensor
            # else:
            #     input_state = next_whole_state[:, -virleader_obs_len:]
            #     next_action = agent.functor_dict['actor_target'](input_state)  # 100*2
            #     next_whole_action.append(next_action)  # 包含 所有agent一个batch的action ,4个 100*2 tensor
        # 初始化结果张量
        concatenated_tensor = next_whole_action[0]
        # 循环按列拼接张量
        for i in range(1, len(next_whole_action)):
            concatenated_tensor = torch.cat([concatenated_tensor, next_whole_action[i]], dim=1) # 将torch框架下的张量action按列拼接


        target_next_value = agents[num].functor_dict['critic_target'](next_whole_state[:,obs_len*num:obs_len * (num + 1)], concatenated_tensor)
        #print('reward', reward)
        # target_next_value = agents[num].functor_dict['critic_target'](
        #     next_whole_state, concatenated_tensor)
        #print(done_value)
        whole = whole_reward[:,num].reshape(-1,1) # 第num个agent的奖励值
        whole1 = whole_done_value[:,num].reshape(-1,1)#第num个agent的done值
        target_value = whole + gamma * whole1 * target_next_value
    # 计算平方差并求平均
    td_error = target_value - value
    value_loss = torch.sum(weight * torch.square(td_error)) / len(value)
        # noise_range = 0.05  # 噪声范围为目标值的5%
        # # 生成随机噪声
        # noise = np.random.uniform(low=-noise_range * target_value, high=noise_range * target_value)
        # # 添加噪声
        # noise = noise.astype(np.float32)
        # target_value += noise

    # value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agents[num].optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agents[num].optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss,td_error
def critic_updator_ddpg_localstate_virleader(env,agents,num, whole_state,whole_action,whole_reward, next_whole_state, whole_done_value, gamma, ):
    """
    含有virleader系统的所有critic的更新,采用局部状态，局部动作
    """

    obs_space = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_space[0]
    virleader_obs_len = virleader_obs_space[0]
    action_dim = get_action_dim(env.action_space)
    next_whole_action = []
    purnum = env.purnum
    # if num != purnum-1:
    value = agents[num].functor_dict['critic'](whole_state[:, obs_len * num:obs_len * (num + 1)], whole_action[:,2*num:2*(num+1)])
    with torch.no_grad():
        input_state = next_whole_state[:, obs_len * num:obs_len * (num + 1)]
        next_action = agents[num].functor_dict['actor_target'](input_state)
        target_next_value = agents[num].functor_dict['critic_target'](next_whole_state[:, obs_len * num:obs_len * (num + 1)], next_action)
    # else:
    #     value = agents[num].functor_dict['critic'](whole_state[:, -virleader_obs_len:],
    #                                                whole_action[:, 2 * num:2 * (num + 1)])
    #     with torch.no_grad():
    #         input_state = next_whole_state [:, -virleader_obs_len:]
    #         next_action = agents[num].functor_dict['actor_target'](input_state)
    #         target_next_value = agents[num].functor_dict['critic_target'](
    #             next_whole_state[:, -virleader_obs_len:], next_action)
    whole = whole_reward[:,num].reshape(-1,1) # 第num个agent的奖励值
    whole1 = whole_done_value[:,num].reshape(-1,1)#第num个agent的done值
    target_value = whole + gamma * whole1 * target_next_value
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agents[num].optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agents[num].optimizer_dict['critic'].step()

    return value_loss


def actor_updator_ddpg_localstate_virleader(env, agents, num, whole_state):
    """
       含有virleader系统的所有actor的更新，采用局部状态
       """
    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = False
    obs_shape = get_obs_shape(env.observation_space)
    obs_len = obs_shape[0]
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    virleader_obs_len = virleader_obs_space[0]
    # action_dim = get_action_dim(env.action_space)

    # if num != env.purnum-1:
    action = agents[num].functor_dict['actor'](whole_state[:, obs_len * num:obs_len * (num + 1)])
    policy_loss = - agents[num].functor_dict['critic'](whole_state[:, obs_len * num:obs_len * (num + 1)], action).mean()

    # else:
    #     action = agents[num].functor_dict['actor'](whole_state[:, -virleader_obs_len:])
    #     policy_loss = - agents[num].functor_dict['critic'](whole_state[:, -virleader_obs_len:],
    #                                                        action).mean()


    agents[num].optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agents[num].optimizer_dict['actor'].step()

    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss
def critic_updator_ddpg_wholestate(env,agents,num, whole_state,whole_action,whole_reward, next_whole_state, whole_done_value, gamma, ):
    """所有agent状态，动作训练critic， 单个agent的状态训练actor"""
    value = agents[num].functor_dict['critic'](whole_state, whole_action)
    obs_space = get_obs_shape(env.observation_space)
    obs_len = obs_space[0]
    next_whole_action = []
    with torch.no_grad():
        for i, agent in enumerate(agents):

            input_state = next_whole_state[:, obs_len * i:obs_len * (i + 1)]
            next_action = agent.functor_dict['actor_target'](input_state) # 100*2
            next_whole_action.append(next_action) # 包含 所有agent一个batch的action ,4个 100*2 tensor

        # 初始化结果张量
        concatenated_tensor = next_whole_action[0]
        # 循环按列拼接张量
        for i in range(1, len(next_whole_action)):
            concatenated_tensor = torch.cat([concatenated_tensor, next_whole_action[i]], dim=1) # 将torch框架下的张量action按列拼接


        target_next_value = agents[num].functor_dict['critic_target'](next_whole_state, concatenated_tensor)
        #print('reward', reward)
        #print(done_value)
        whole = whole_reward[:,num].reshape(-1,1) # 第num个agent的奖励值
        whole1 = whole_done_value[:,num].reshape(-1,1)#第num个agent的done值
        target_value = whole + gamma * whole1 * target_next_value
        # noise_range = 0.05  # 噪声范围为目标值的5%
        # # 生成随机噪声
        # noise = np.random.uniform(low=-noise_range * target_value, high=noise_range * target_value)
        # # 添加噪声
        # noise = noise.astype(np.float32)
        # target_value += noise

    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agents[num].optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agents[num].optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss
def actor_updator_maddpg(env, agents, num, whole_state):
    """critic 输入所有状态和动作，actor输入自身状态和动作"""
    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = False
    obs_shape = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_shape[0]
    # action_dim = get_action_dim(env.action_space)
    whole_action = []
    virleader_obs_len = virleader_obs_space[0]
    purnum = env.purnum
    for i, agent in enumerate(agents):
        if i!=env.purnum-1:
            action = agent.functor_dict['actor'](whole_state[:, obs_len * i:obs_len * (i + 1)])
            whole_action.append(action)
    # 初始化结果张量
    concatenated_tensor = whole_action[0]
    # 循环按列拼接张量
    for i in range(1, len(whole_action)):
        concatenated_tensor = torch.cat([concatenated_tensor, whole_action[i]], dim=1)  # 将张量action按列拼接
    policy_loss = - agents[num].functor_dict['critic'](whole_state[:, :-virleader_obs_len], concatenated_tensor).mean()
    # print(policy_loss)
    agents[num].optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agents[num].optimizer_dict['actor'].step()

    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss
def actor_updator_maddpg_1(env, agents, num, whole_state):
    """critic 输入所有状态和动作，actor输入自身状态和动作"""
    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = False
    obs_shape = get_obs_shape(env.observation_space)
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    # virleader_obs_len = virleader_obs_space[0]
    obs_len = obs_shape[0]
    # action_dim = get_action_dim(env.action_space)
    whole_action = []
    virleader_obs_len = virleader_obs_space[0]
    purnum = env.purnum
    for i, agent in enumerate(agents):
        if i!=env.purnum-1:
            action = agent.functor_dict['actor'](whole_state[:, obs_len * i:obs_len * (i + 1)])
        else:
            action = agent.functor_dict['actor'](whole_state[:,-virleader_obs_len:])
        whole_action.append(action)
    # 初始化结果张量
    concatenated_tensor = whole_action[0]
    # 循环按列拼接张量
    for i in range(1, len(whole_action)):
        concatenated_tensor = torch.cat([concatenated_tensor, whole_action[i]], dim=1)  # 将张量action按列拼接
    policy_loss = - agents[num].functor_dict['critic'](whole_state[:, obs_len * num:obs_len * (num + 1)], concatenated_tensor).mean()
    # policy_loss = - agents[num].functor_dict['critic'](whole_state,
    #                                                    concatenated_tensor).mean()
    # print(policy_loss)
    agents[num].optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agents[num].optimizer_dict['actor'].step()

    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss
# virleader 状态与其它agent状态不一样
def actor_updator_ddpg_wholestate_virleader(env, agents, num, whole_state):
    """
       含有virleader系统的所有actor的更新
       """
    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = False
    obs_shape = get_obs_shape(env.observation_space)
    obs_len = obs_shape[0]
    virleader_obs_space = get_obs_shape(env.virleader_observation_space)
    virleader_obs_len = virleader_obs_space[0]
    # action_dim = get_action_dim(env.action_space)
    whole_action = []
    purnum = env.purnum
    for i, agent in enumerate(agents):
        if i != purnum-1:
            action = agent.functor_dict['actor'](whole_state[:, obs_len * i:obs_len * (i + 1)])
            whole_action.append(action)
        else:
            action = agent.functor_dict['actor'](whole_state[:, -virleader_obs_len:])
            whole_action.append(action)
    # 初始化结果张量
    concatenated_tensor = whole_action[0]
    # 循环按列拼接张量
    for i in range(1, len(whole_action)):
        concatenated_tensor = torch.cat([concatenated_tensor, whole_action[i]], dim=1)  # 将张量action按列拼接
    policy_loss = - agents[num].functor_dict['critic'](whole_state, concatenated_tensor).mean()
    # print(policy_loss)
    agents[num].optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agents[num].optimizer_dict['actor'].step()

    for p in agents[num].functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss

# virleader actor更新
def virleader_actor_updator_ddpg(env,agent, state,):
    """
       含有virleader系统的virleader 的actor的更新
       """
    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = False
    obs_shape = get_obs_shape(env.virleader_observation_space)
    obs_len = obs_shape[0]
    action_dim = get_action_dim(env.action_space)
    policy_loss = - agent.functor_dict['critic'](state[:,-obs_len:], agent.functor_dict['actor'](state[:,-obs_len:])).mean()
    # print(policy_loss)
    agent.optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agent.optimizer_dict['actor'].step()

    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss
def actor_updator_ddpg(agent, state, action, reward, next_state):
    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = False

    policy_loss = - agent.functor_dict['critic'](state, agent.functor_dict['actor'](state)).mean()
    # print(policy_loss)
    agent.optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agent.optimizer_dict['actor'].step()

    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss


def discriminator_updator(agent, state, action, label):
    '''
    judge data whether from actor or demonstrator.
    loss = CrossEntropy
    :param agent: agent object
    :param state:
    :param action:
    :param label:
    :return:
    '''
    x = state
    # print('x',x[0])
    # print('a',action[0])
    # print('l',label[0])
    for p in agent.discriminator.parameters():
        p.requires_grad = True
    z = agent.discriminator(x, action)
    #print(z.shape)

    y = torch.log(z)
    #print(y)

    loss = nn.NLLLoss()(y, label.reshape(-1).long())
    #print(loss)
    #print(agent.optimizer_discriminator)
    agent.optimizer_discriminator.zero_grad()
    loss.backward()
    # for name, parms in agent.discriminator.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:',torch.norm(parms.grad))
    agent.optimizer_discriminator.step()

    return

def actor_updator_gail(agent,state):
    '''
    update actor, the Value (Critic) is discriminator.
    :param agent:
    :param state:
    :return:
    '''
    for p in agent.discriminator.parameters():
        p.requires_grad = False

    for p in agent.actor.parameters():
        p.requires_grad = True

    policy_loss = - agent.discriminator(state, agent.actor(state))[:, 1].mean()

    #print(policy_loss)
    #print(agent.discriminator)
    agent.optimizer_actor.zero_grad()
    policy_loss.backward()

    agent.optimizer_actor.step()

    return policy_loss

def critic_updator_dqn(agent, state, action, reward, next_state, done_value, gamma, ):
    #print(action.shape)
    value = torch.gather(agent.functor_dict['critic'](state), dim=1, index=action.long())
    #print(value.shape)
    with torch.no_grad():
        next_action = torch.argmax(agent.functor_dict['critic'](next_state), dim=1).reshape((-1, 1))

        target_next_value = torch.gather(agent.functor_dict['critic'](next_state), dim=1, index=next_action.long())
        # print('reward', reward.shape)
        target_value = reward + gamma * (done_value) * target_next_value
    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value.mean(),)
    # print('value', value.mean())
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss


def critic_updator_sac(agent, obs, action, reward, next_obs, not_done, gamma):
    with torch.no_grad():
        #print('nt', next_obs)
        _, policy_action, log_pi, _ = agent.functor_dict['actor'](next_obs)
        #print('no', next_obs)
        target_Q1, target_Q2 = agent.functor_dict['critic_target'](next_obs, policy_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - agent.functor_dict['log_alpha'].exp().detach() * log_pi
        target_Q = reward + (not_done * gamma * target_V)

    # get current Q estimates
    current_Q1, current_Q2 = agent.functor_dict['critic'](
        obs, action)
    critic_loss = F.mse_loss(current_Q1,
                             target_Q) + F.mse_loss(current_Q2, target_Q)



    # Optimize the critic
    agent.optimizer_dict['critic'].zero_grad()
    critic_loss.backward()
    agent.optimizer_dict['critic'].step()

    # for name, parms in agent.functor_dict['critic'].named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)


def actor_and_alpha_updator_sac(agent, obs, target_entropy):
    # detach encoder, so we don't update it with the actor loss
    _, pi, log_pi, log_std = agent.functor_dict['actor'](obs)
    actor_Q1, actor_Q2 = agent.functor_dict['critic'](obs, pi)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (agent.functor_dict['log_alpha'].exp().detach() * log_pi - actor_Q).mean()

    entropy = 0.5 * log_std.shape[1] * \
        (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)


    # optimize the actor
    agent.optimizer_dict['actor'].zero_grad()
    actor_loss.backward()
    agent.optimizer_dict['actor'].step()


    agent.optimizer_dict['log_alpha'].zero_grad()
    alpha_loss = (agent.functor_dict['log_alpha'].exp() *
                  (-log_pi - target_entropy).detach()).mean()

    alpha_loss.backward()
    agent.optimizer_dict['log_alpha'].step()
