import os

import click
import numpy as np
import pandas as pd

from es_distributed.policies import BaselinePolicy
from util import sonic_util
from util.retro_registry import register_all
import gym
@click.command()
@click.argument('train_level')
@click.argument('policy_file')
@click.argument('storage')
@click.option('--include_test', is_flag=True)
@click.option('--record', is_flag=True)

@click.option('--extra_kwargs')

def main(train_level, policy_file, storage, include_test, record, extra_kwargs):
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

    to_evaluate.sort()
    train_perf = []
    test_perf = []
    # how often to repeat scoring
    n_rep = 1
    # check every nth policy on the test set
    test_eval_interval = 4

    counter = 0
    # evaluate on this level first
    for policy_file_path in to_evaluate:
        print('Scoring ' + policy_file_path)
        all_scores, all_percs, all_lengths, all_rewards = evaluate_policy_on_levels(policy_file_path,[train_level],n_rep=n_rep, record=record)
        # prompt user to find out whether to evaluate only first policy
        if counter == 0:
            cont = input('Continue? y / n: ')
            if cont is not 'y':
                return
        perf =[all_scores,all_percs,all_lengths,all_rewards]
        train_perf.append(perf)
        # if test set performance is to be measured too
        if(include_test and (counter % test_eval_interval == 0 or counter + 1 == len(to_evaluate))):
            all_scores, all_percs, all_lengths, all_rewards = evaluate_policy_on_test_set(policy_file_path,n_rep,record=record)
            perf = [np.mean(all_scores), np.mean(all_percs), np.mean(all_lengths), np.mean(all_rewards)]
            test_perf.append(perf)
        counter = counter + 1

    train_perf_frame = pd.DataFrame(train_perf, columns=['score', 'perc','length','reward'])


    train_perf_frame.to_csv('train_' + storage + '.csv')

    if(include_test):
        test_perf_frame = pd.DataFrame(test_perf, columns=['mean_score', 'mean_perc', 'mean_length', 'mean_reward'])
        test_perf_frame.to_csv('test_' + storage + '.csv')

from gym import wrappers


def evaluate_policy_on_test_set(policy_file,n_rep,record=False):
    ids = []
    for count in range(1, 11):
        ids.append('Test-v' + str(count))

    return evaluate_policy_on_levels(policy_file,ids,n_rep,record=record)

def evaluate_policy_on_levels(policy_file,ids,n_rep, record= False):

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
        # create a baseline policy
        if policy_file == 'baseline':
            pi = BaselinePolicy()
        else:
            # load the policy just once
            pi = GAGenesisPolicy.Load(policy_file)
        # load the policy just once
        # play each env
        for id in ids:
            env = make_env(id,record=record)

            temp_all_scores = []
            temp_all_lengths = []
            temp_all_percs = []
            temp_all_rewards = []

            for i in range(0,n_rep):

                if pi.needs_ref_batch:
                    pi.set_ref_batch(get_ref_batch(env, batch_size=128))
                # play on this env
                rews, t, res_dict = pi.rollout(env, render=False)
                temp_all_lengths.append(t)
                # store the list of rewards
                perc = res_dict['max_perc'] if 'max_perc' in res_dict else 0
                temp_all_percs.append(perc)
                score = res_dict['max_score'] if 'max_score' in res_dict else 0
                temp_all_scores.append(score)
                temp_all_rewards.append(rews.sum())
            del env

            print(temp_all_percs)

            all_scores.append(np.mean(temp_all_scores))
            all_rewards.append(np.mean(temp_all_rewards))
            all_lengths.append(np.mean(temp_all_lengths))
            all_percs.append(np.mean(temp_all_percs))

    # scores, how far Sonic got to the goal, level length, and in-game score
    if len(all_scores) == 1:
        return all_scores[0], all_percs[0], all_lengths[0], all_rewards[0]
    else:
        return all_scores, all_percs, all_lengths, all_rewards

def make_env(env_id, record=True, extra_kwargs=None):
    env = gym.make(env_id)
    env = sonic_util.sonicize_env(env)

    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/results/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    return env


if __name__ == '__main__':
    main()
