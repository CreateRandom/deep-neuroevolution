import sys
import time
import numpy as np
import losswise
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import scripts.run_comp as run_comp
from util.retro_registry import register_all


class MyHandler(PatternMatchingEventHandler):


    def __init__(self, patterns=None, ignore_patterns=None, ignore_directories=False, case_sensitive=False):
        super().__init__(patterns, ignore_patterns, ignore_directories, case_sensitive)
        losswise.set_api_key('W5BWWF7RP')  # api_key for 'NC'
        # tracking stuff
        session = losswise.Session(tag='test_set_performance')
        self.graph = session.graph('reward', kind='max')
        self.it_counter = 0

        max_episode_steps = 4500
        register_all(max_episode_steps=max_episode_steps)

    # watch only for model files
    patterns = ["*.h5"]

    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        # evaluate the policy11
        print(event.src_path)  # print now only for debug
        runner = run_comp.CompRunner()
        rewards, times = runner.evaluate_policy(event.src_path)
        del runner

        rew_dict = {}
        counter = 1
        for reward in rewards:
            rew_dict['level' + str(counter)] = reward
            counter = counter + 1
        self.graph.append(self.it_counter,rew_dict)
        self.graph.append(self.it_counter,{'mean':np.mean(rewards)})
        self.it_counter = self.it_counter + 1
        print(rewards)

    # only when file was created
    def on_created(self, event):
        self.process(event)

# watches for new h5 policy files on the path and evaluates them on the test
# set
if __name__ == '__main__':
    args = sys.argv[1:]
    observer = Observer()
    observer.schedule(MyHandler(), path=args[0] if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()