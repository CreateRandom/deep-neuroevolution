import os

import click
import numpy as np
import pandas as pd

from util import sonic_util
from util.retro_registry import register_all
import gym
@click.command()
@click.argument('train_level')
@click.argument('policy_file')

@click.option('--include_test', is_flag=True)
@click.option('--extra_kwargs')

def main(train_level, policy_file, include_test, extra_kwargs):
    max_episode_steps = 4500
    # register retro games with max steps per episode to be played out
    register_all(max_episode_steps=max_episode_steps)

    to_evaluate = []
    # first find out whether policy_file is a path / a single file
    if(os.path.isdir(policy_file)):
        for dirpath, subdirs, files in os.walk(policy_file):
            for x in files:
                if x.endswith(".h5"):
                    to_evaluate.append(os.path.join(dirpath, x))
    else:
        to_evaluate.append(policy_file)

    train_perf = []
    test_perf = []


    # evaluate on this level first
    for policy_file_path in to_evaluate:
        print('Scoring ' + policy_file_path)
        all_scores, all_percs, all_lengths, all_rewards = evaluate_policy_on_levels(policy_file_path,[train_level])
        perf =[all_scores,all_percs,all_lengths,all_rewards]
        train_perf.append(perf)
        # if test set performance is to be measured too
        if(include_test):
            all_scores, all_percs, all_lengths, all_rewards = evaluate_policy_on_test_set(policy_file_path)
            perf = [np.mean(all_scores), np.mean(all_percs), np.mean(all_lengths), np.mean(all_rewards)]
            test_perf.append(perf)

    train_perf_frame = pd.DataFrame(train_perf, columns=['score', 'perc','length','reward'])


    train_perf_frame.to_csv('train.csv')

    if(include_test):
        test_perf_frame = pd.DataFrame(test_perf, columns=['mean_score', 'mean_perc', 'mean_length', 'mean_reward'])
        test_perf_frame.to_csv('test.csv')

from gym import wrappers


def evaluate_policy_on_test_set(policy_file):
    ids = []
    for count in range(1, 11):
        ids.append('Test-v' + str(count))

    return evaluate_policy_on_levels(policy_file,ids)

def evaluate_policy_on_levels(policy_file,ids):

    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy, ESAtariPolicy, GAAtariPolicy, GAGenesisPolicy
    from es_distributed.atari_wrappers import ScaledFloatFrame, wrap_deepmind
    from es_distributed.es import get_ref_batch
    import numpy as np

    is_atari_policy = True

    all_scores = []
    all_lengths = []
    all_percs = []
    all_rewards = []

    tf.reset_default_graph()

    with tf.Session():
        # load the policy just once
        pi = GAGenesisPolicy.Load(policy_file)
        # load the policy just once
        # play each env
        for id in ids:
            env = make_env(id)

            if pi.needs_ref_batch:
                pi.set_ref_batch(get_ref_batch(env, batch_size=128))
            #  while total_steps < max_steps_per_level:
            # play on this env
            rews, t, res_dict = pi.rollout(env, render=False)
            all_lengths.append(t)
            # store the list of rewards
            perc = res_dict['max_perc'] if 'max_perc' in res_dict else 0
            all_percs.append(perc)
            score = res_dict['max_score'] if 'max_score' in res_dict else 0

            all_scores.append(score)
            all_rewards.append(rews.sum())
            del env

    # scores, how far Sonic got to the goal, level length, and in-game score
    if len(all_scores) == 1:
        return all_scores[0], all_percs[0], all_lengths[0], all_rewards[0]
    else:
        return all_scores, all_percs, all_lengths, all_rewards

def make_env(env_id, record=False, extra_kwargs=None):
    env = gym.make(env_id)
    env = sonic_util.sonicize_env(env)

    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    return env


if __name__ == '__main__':
    main()
