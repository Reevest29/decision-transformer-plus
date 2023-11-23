import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class AutoEncoderTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        reward_target = torch.clone(rewards)

        states_shape, actions_shape, rewards_shape = states.shape, actions.shape, rewards.shape

        batch_size, context_len, _ = states_shape
        state_dim = states_shape[-1]
        action_dim = actions_shape[-1]
        reward_dim = rewards_shape[-1]

        flat_states = states.reshape(-1, state_dim)
        flat_actions = actions.reshape(-1, action_dim)
        flat_rewards = rewards.reshape(-1, reward_dim)


        num_tokens = batch_size * context_len
        drop_prob = torch.random(0.7,0.9)
        state_idxs = torch.randint(high=num_tokens,size=(int(num_tokens*drop_prob),))
        action_idxs = torch.randint(high=num_tokens,size=(int(num_tokens*drop_prob),))
        reward_idxs = torch.randint(high=num_tokens,size=(int(num_tokens*drop_prob),))

        flat_states[state_idxs] = torch.zeros(state_dim)
        flat_actions[action_idxs] = torch.zeros(action_dim)
        flat_rewards[reward_idxs] = torch.zeros(reward_dim)

        masked_states = flat_states.reshape(batch_size,context_len,state_dim)
        masked_actions = flat_actions.reshape(batch_size,context_len,action_dim)
        masked_rewards = flat_rewards.reshape(batch_size,context_len,reward_dim)




        state_preds, action_preds, reward_preds = self.model.forward(
            masked_states, masked_actions, masked_rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]

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
