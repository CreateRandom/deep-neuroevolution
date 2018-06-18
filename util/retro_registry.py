from gym.envs.registration import register

# all states used as the test set, a total of 10 states
test_state_dict = {
    'SonicTheHedgehog-Genesis': [
        'SpringYardZone.Act1','GreenHillZone.Act2',
        'StarLightZone.Act3', 'ScrapBrainZone.Act1'],
    'SonicTheHedgehog2-Genesis': [
        'MetropolisZone.Act3', 'HillTopZone.Act2', 'CasinoNightZone.Act2'],
    'SonicAndKnuckles3-Genesis': ['LavaReefZone.Act1',
                                  'FlyingBatteryZone.Act2',
                                  'HydrocityZone.Act1',
                                  'AngelIslandZone.Act2']
}

# to use gym environments via gym.make, they have to be registered first
# this method takes care of that by registering all games and levels
# a total of 17 + 18 + 23 = 58 available levels

all_state_dict = {
    'SonicTheHedgehog-Genesis': ['GreenHillZone.Act1', 'GreenHillZone.Act2', 'GreenHillZone.Act3',
                                 'LabyrinthZone.Act1', 'LabyrinthZone.Act2', 'LabyrinthZone.Act3',
                                 'MarbleZone.Act1', 'MarbleZone.Act2', 'MarbleZone.Act3',
                                 'ScrapBrainZone.Act1', 'ScrapBrainZone.Act2',
                                 'SpringYardZone.Act1', 'SpringYardZone.Act2', 'SpringYardZone.Act3',
                                 'StarLightZone.Act1', 'StarLightZone.Act2', 'StarLightZone.Act3'],
    'SonicTheHedgehog2-Genesis': ['AquaticRuinZone.Act1', 'AquaticRuinZone.Act2',
                                   'CasinoNightZone.Act1', 'CasinoNightZone.Act2',
                                   'ChemicalPlantZone.Act1', 'ChemicalPlantZone.Act2',
                                   'EmeraldHillZone.Act1', 'EmeraldHillZone.Act2',
                                   'HillTopZone.Act1', 'HillTopZone.Act2',
                                   'MetropolisZone.Act1', 'MetropolisZone.Act2', 'MetropolisZone.Act3',
                                   'MysticCaveZone.Act1', 'MysticCaveZone.Act2',
                                   'OilOceanZone.Act1', 'OilOceanZone.Act2',
                                   'WingFortressZone'],
    'SonicAndKnuckles3-Genesis': ['AngelIslandZone.Act1', 'AngelIslandZone.Act2',
                                  'CarnivalNightZone.Act1', 'CarnivalNightZone.Act2',
                                  'DeathEggZone.Act1', 'DeathEggZone.Act2',
                                  'FlyingBatteryZone.Act1', 'FlyingBatteryZone.Act2',
                                  'HiddenPalaceZone',
                                  'HydrocityZone.Act1', 'HydrocityZone.Act2',
                                  'IcecapZone.Act1', 'IcecapZone.Act2',
                                  'LaunchBaseZone.Act1', 'LaunchBaseZone.Act2',
                                  'LavaReefZone.Act1', 'LavaReefZone.Act2',
                                  'MarbleGardenZone.Act1', 'MarbleGardenZone.Act2',
                                  'MushroomHillZone.Act1', 'MushroomHillZone.Act2',
                                  'SandopolisZone.Act1', 'SandopolisZone.Act2']
}

# importing retro here causes problems with tensorflow down the road
# for unclear reasons, so instead of importing it, just list all the
# available states manually for now

# TODO: add other games and other states (i.e. game levels to make them usable)
def register_all(max_episode_steps):

    # remove the test levels for all games
    for game in all_state_dict.keys():
        for level in test_state_dict[game]:
            all_state_dict[game].remove(level)

    register(id='Sonic1-v0',
             entry_point='util.MultiStateEnv:MultiStateEnv',
             kwargs={'game': 'SonicTheHedgehog-Genesis', 'states' : all_state_dict['SonicTheHedgehog-Genesis']},
             max_episode_steps=max_episode_steps
             )

    register(id='Sonic2-v0',
             entry_point='util.MultiStateEnv:MultiStateEnv',
             kwargs={'game': 'SonicTheHedgehog2-Genesis', 'states': all_state_dict['SonicTheHedgehog2-Genesis']},
             max_episode_steps=max_episode_steps
             )

    register(id='Sonic3-v0',
             entry_point='util.MultiStateEnv:MultiStateEnv',
             kwargs={'game': 'SonicAndKnuckles3-Genesis', 'states': all_state_dict['SonicAndKnuckles3-Genesis']},
             max_episode_steps=max_episode_steps
             )


    # register all test levels
    count = 1
    for game in test_state_dict.keys():
        #for level in retro.list_states(game):
        for level in test_state_dict[game]:

            id = 'Test-v' + str(count)
            register(id=id,
                     entry_point='retro.retro_env:RetroEnv',
                     kwargs={'game': game, 'state': level},
                     max_episode_steps=max_episode_steps
                    )
            count = count + 1

    # register all test levels
    count = 1
   # game_list = ['SonicTheHedgehog-Genesis', 'SonicTheHedgehog2-Genesis','SonicAndKnuckles3-Genesis']
    for game in all_state_dict.keys():
        print(game)
        #for level in retro.list_states(game):
        for level in all_state_dict[game]:

            id = 'SingleSonic-v' + str(count)
            register(id=id,
                     entry_point='retro.retro_env:RetroEnv',
                     kwargs={'game': game, 'state': level},
                     max_episode_steps=max_episode_steps
                    )
            count = count + 1

    # # test register a retro game
    # register(id='Sonic1-1-v0',
    #          entry_point='retro.retro_env:RetroEnv',
    #          kwargs={'game': 'SonicTheHedgehog-Genesis'},
    #          max_episode_steps=max_episode_steps
    #          )