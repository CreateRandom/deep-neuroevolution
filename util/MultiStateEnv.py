import gzip
import json
import os
import random
import sys

import retro

from retro.retro_env import RetroEnv
import numpy as np

# idea: have multiple initial states (one per game level) and reset
# to a different one after reset was called a number of times
# to allow different levels to be accessed within one game
class MultiStateEnv(RetroEnv):
    import retro


    def __init__(self, game, states, scenario=None, info=None,
                 use_restricted_actions=retro.ACTIONS_FILTERED, record=False):
        super().__init__(game, retro.STATE_NONE, scenario, info, use_restricted_actions, record)
        # path loading
        self.game_path = retro.get_game_path(game)
        self.metadata_path = os.path.join(self.game_path, 'metadata.json')

        # list of states --> levels
        self.initial_states = {}

        for state_id in states:
            self.initial_states[state_id] = self.read_state(state_id)


    def reset(self):

        # pick an initial state at random to reset to
        pick = random.choice(list(self.initial_states.keys()))
        self.em.set_state(self.initial_states[pick])

        # copied over
        self.em.set_button_mask(np.zeros([16], np.uint8))
        self.em.step()
        if self.movie_path is not None:
            self.record_movie(os.path.join(self.movie_path, '%s-%s-%04d.bk2' % (self.gamename, self.statename, self.movie_id)))
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.img = ob = self.em.get_screen()
        self.data.reset()
        self.data.update_ram()
        return ob


    # return a state rep that can be used as the initial state
    def read_state(self, state):

        import retro
        if state == retro.STATE_NONE:
            return None
        elif state == retro.STATE_DEFAULT:
            try:
                with open(self.metadata_path) as f:
                    metadata = json.load(f)
                if 'default_state' in metadata:
                    with gzip.open(os.path.join(self.game_path, metadata['default_state']) + '.state', 'rb') as fh:
                        return fh.read()
            except (IOError, json.JSONDecodeError):
                pass
        else:
            if not state.endswith('.state'):
                state += '.state'

            with gzip.open(os.path.join(self.game_path, state), 'rb') as fh:
                return fh.read()