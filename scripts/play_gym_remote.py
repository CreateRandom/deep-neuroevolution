import gym_remote.exceptions as gre
import gym_remote.client as grc

from click._unicodefun import click

# this is a sample function that loads a trained
# agent from an h5 file and then runs it via the virtual environment
# it can be submitted in a docker container

@click.command()
@click.argument('policy_file')
@click.option('--extra_kwargs')
def main(policy_file,extra_kwargs):
    import tensorflow as tf
    from es_distributed.policies import GAGenesisPolicy

    print('connecting to remote environment')
    env = grc.RemoteEnv('tmp/sock')
    print('starting episode')
    ob = env.reset()

    with tf.Session():
        pi = GAGenesisPolicy.Load(policy_file, extra_kwargs=extra_kwargs)

        while True:
            # act
            action = pi.act(ob[None])
            # and step through
            ob, reward, done, _ = env.step(action[0])
            env.render()
            if done:
                print('episode complete')
                env.reset()



if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)