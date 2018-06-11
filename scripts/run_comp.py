import click

from util import sonic_util
from util.retro_registry import register_all
import gym
from gym import wrappers

@click.command()
@click.argument('policy_file')
@click.option('--record', is_flag=False)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')

def main(policy_file, record, stochastic, extra_kwargs):
    return evaluate_policy(policy_file)


def evaluate_policy(policy_file):

    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy, ESAtariPolicy, GAAtariPolicy, GAGenesisPolicy
    from es_distributed.atari_wrappers import ScaledFloatFrame, wrap_deepmind
    from es_distributed.es import get_ref_batch
    import numpy as np
    max_episode_steps = 4500
    # register retro games with max steps per episode to be played out
    register_all(max_episode_steps=max_episode_steps)

    is_atari_policy = True

    all_rewards = []
    all_lengths = []

    with tf.Session():
        # load the policy just once
        pi = GAGenesisPolicy.Load(policy_file)
        # load the policy just once
        # play each env
        for count in range(1, 10):
            id = 'Test-v' + str(count)
            env = make_env(id)

            if pi.needs_ref_batch:
                pi.set_ref_batch(get_ref_batch(env, batch_size=128))
            #  while total_steps < max_steps_per_level:
            # play on this env
            rews, t, novelty_vector = pi.rollout(env, render=False)
            all_lengths.append(t)
            # store the list of rewards
            all_rewards.append(rews.sum())
            del env

        # save rewards

    return all_rewards, all_lengths

def make_env(env_id, record=False, extra_kwargs=None):
    env = gym.make(env_id)
    env = sonic_util.SonicDiscretizer(env)

    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    return env


if __name__ == '__main__':
    main()
