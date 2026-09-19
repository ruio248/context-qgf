"""Native QGF continuation control with a frozen flow actor."""

from __future__ import annotations

import jax
import ml_collections

from agents.qgf import QGFAgent
from agents.qgf import get_config as get_qgf_config
from utils.flax_utils import target_update


class QGFQVFinetuneAgent(QGFAgent):
    """Continue native Q/V updates while holding the behavior actor fixed."""

    @jax.jit
    def update(self, batch):
        new_rng, _ = jax.random.split(self.rng, 2)
        new_critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda params: self.critic_loss(batch, params)
        )
        new_target_critic = target_update(
            new_critic, self.target_critic, self.config["tau"]
        )
        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda params: self.value_loss(batch, params)
        )
        return self.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
        ), {**critic_info, **value_info}


def get_config():
    config = get_qgf_config()
    config["agent_name"] = "qgf_qv_finetune"
    return ml_collections.ConfigDict(config)
