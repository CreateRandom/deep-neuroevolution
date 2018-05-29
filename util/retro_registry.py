from gym.envs.registration import register


# to use gym environments via gym.make, they have to be registered first
# this method takes care of that by registering all games and levels

# all the retro ids of the games we can sample levels from
game_ids = ['SonicTheHedgehog-Genesis',
            'SonicTheHedgehog2-Genesis',
            'SonicAndKnuckles3-Genesis']

# TODO: add other games and other states (i.e. game levels to make them usable)
def register_all(max_episode_steps):
    # test register a retro game
    register(id='Sonic1-v0',
             entry_point='retro.retro_env:RetroEnv',
             kwargs={'game': 'SonicTheHedgehog-Genesis'},
             max_episode_steps=max_episode_steps
             )