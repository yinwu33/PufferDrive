from __future__ import annotations

from pathlib import Path

import torch


class PufferModule:
    def __init__(self, config, env, model_path, deterministic=False, device=None):
        import pufferlib.ocean.torch as ocean_torch

        self.env = env
        self.deterministic = deterministic
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and not deterministic else "cpu"
        )

        policy_cls = getattr(ocean_torch, config.get("base", {}).get("policy_name", "Drive"))
        policy = policy_cls(env, **config.get("policy", {}))

        rnn_name = config.get("base", {}).get("rnn_name")
        self.recurrent = rnn_name is not None
        if self.recurrent:
            rnn_cls = getattr(ocean_torch, rnn_name)
            policy = rnn_cls(env, policy, **config.get("rnn", {}))

        state_dict = torch.load(
            Path(model_path), map_location=self.device, weights_only=True
        )
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
        policy.load_state_dict(state_dict)
        policy.to(self.device)
        policy.eval()

        self.policy = policy
        self.lstm_state = None
        if self.recurrent:
            self.reset_state(env.num_agents)

    def reset_state(self, num_agents=None):
        if not self.recurrent:
            return
        num_agents = num_agents or self.env.num_agents
        self.lstm_state = {
            "lstm_h": torch.zeros(
                num_agents, self.policy.hidden_size, device=self.device
            ),
            "lstm_c": torch.zeros(
                num_agents, self.policy.hidden_size, device=self.device
            ),
        }

    def step(self, obs):
        import pufferlib.pytorch

        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if self.recurrent and self.lstm_state["lstm_h"].shape[0] != obs.shape[0]:
                self.reset_state(obs.shape[0])
            logits, _ = self.policy.forward_eval(
                obs, self.lstm_state if self.recurrent else None
            )
            if self.deterministic:
                action = pufferlib.pytorch.deterministic_action(logits)
            else:
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
        return action.cpu().numpy()
