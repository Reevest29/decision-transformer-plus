import numpy as np
import torch

import time
from tqdm import tqdm
from copy import deepcopy
class PreTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch, get_trajectory, loss_fn, env, max_ep_len, 
                 pretrain_steps, pretrain_iters, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.get_trajectory = get_trajectory
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.pretrain_steps = pretrain_steps
        self.pretrain_iters = pretrain_iters
        self.env = env
        self.max_ep_len = max_ep_len
        self.start_time = time.time()
        self.context_len = self.model.max_length

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        if iter_num < self.pretrain_iters:
            print(f"Pretrain Loop #{iter_num}")
            for t in tqdm(range(self.pretrain_steps)):
                train_loss = self.pretrain_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            print(f"Training Loop #{iter_num}")
            for t in tqdm(range(num_steps)):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        print(f"Evaluation Loop #{iter_num}")
        for eval_fn in (self.eval_fns):
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        reward_target = torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        reward_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]
        reward_target = reward_target.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
    
    def pretrain_step(self):
        # traj = self.get_trajectory(0)
        

        # actions = traj['actions']
        # states = traj['observations']
        # gt_state = self.env.reset()
        # import pdb; pdb.set_trace()
        # max_t = len(actions)
        # obs_max = states.max(0)
        # obs_min = states.min(0)
        self.env.reset()
        gt_states = []
        gt_rewards = []
        running_loss = 0
        for t in range(self.max_ep_len):
            
            # #scale
            # gt_state = (gt_state - obs_min) / (obs_max-obs_min)
            # exp_state = (states[t] - obs_min) / (obs_max-obs_min)

            # abs = np.abs(gt_state - exp_state)
            # mag_gt = np.linalg.norm(gt_state)
            # mag_exp = np.linalg.norm(exp_state)
            # state_diff = abs / (0.5*(mag_gt+mag_exp))
            # state_diff_pct = state_diff.mean() * 100

            # print(state_diff_pct,(1/state_diff_pct))

            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_trajectory(0,t,self.context_len)
            
            # end of trajectory
            # if attention_mask.sum() == attention_mask.shape[1]:
            #     break 
        
            action_target = torch.clone(actions)
            state_target = torch.clone(states)
            reward_target = torch.clone(rewards)


            # forward model pass
            state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

            # Simulate export and predicted actions
            practice_env = deepcopy(self.env)    

            exp_action = actions[0,-1].detach().numpy()
            action_pred = action_preds[0,-1].detach().numpy()
            exp_next_state, exp_next_reward, exp_next_done, _ = self.env.step(exp_action)
            dt_next_state, dt_next_reward, dt_next_done, _ = practice_env.step(action_pred)

            # Apply attention masks
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            state_dim = state_preds.shape[2]
            state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
            state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
            

            reward_dim = reward_preds.shape[2]
            reward_preds = reward_preds.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]
            reward_target = reward_target.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]

            state_target[-1] = torch.from_numpy(dt_next_state)
            reward_target[-1] = dt_next_reward

            import pdb;pdb.set_trace()
            # 
        #     loss = self.loss_fn(
        #     state_preds, 0*action_preds, reward_preds,
        #     state_target,  0*action_target, reward_target,
        # )

            loss = ((state_preds - state_target) **2) + ((reward_preds - reward_target) **2)
            loss = loss.mean()
            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if exp_next_done or dt_next_done:
                break

        return running_loss / t
    
    

