# Natural Computing Project
This repo is our fork of Uber's Deep Neuroevolution Code made for a course project in Natural Computing. We adapted the code to run with Genesis Games, performing deep neuroevolution to train policies for Sonic The Hedgehog games. The configuration we used can be found in `./configuration/sonic_ga.json`. 

As outlined in the report, running experiments at any scale requires access to a computing cluster / cloud server with considerable power. Additionally, running this code requires that certain dependencies be installed, so root privileges are necessary. While all Python dependencies are listed in the requirements file, the following system-wide dependencies need to be installed manually:

* redis-server
* tmux
* cmake
* zlib1g-dev

After installing these, set up a new local environment in the env folder and install the requirements. Manually run 
```pip install gym[atari] ``` to install the atari bindings for Gym, then download and install the Sonic ROMs. This can be done by

* Downloading Sonic ROMs (e.g. by buying them on Steam)  and extracting into a new folder
* Going into the folder and running ```sudo python3 -m retro.import .```

To run an experiment, first make relevant parameter choices by editing the JSON configuration file, e.g. pick the population size or training parameters. After that, run the following commands in the main folder of the project:

* ```sudo bash scripts/local_redis_settings.sh ```
* ```sudo nohup bash scripts/local_run_redis.sh > redis_dump.txt 2>&1 ```
* ```sudo nohup bash scripts/local_run_master.sh $algo configurations/sonic_ga.json > exp_dump.txt &```
* ```sudo nohup bash scripts/local_run_workers.sh $algo > exp_dump2.txt &```

These commands start up a redis-server instance and run the experiments with the specified configurations. Nohup is used for moving the commands into the background, such that you can close your terminal connection to the server.

After enough data has been collected,i.e. enough iterations have passed, each storing the best policy on the main path, kill the processes and then go on to analyze the data by using the run_comp script. It is used by invoking:

``` python3 run_comp.py $level $policy_file_or_path $name_of_condition --record --include_test ```

The script runs the stored policies on one level as well as the test set if request and stores the data in two csv files that can be used for further analyses.



# Original repo documentation

## AI Labs Neuroevolution Algorithms

This repo contains distributed implementations of the algorithms described in:

[1] [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567)

[2] [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/abs/1712.06560)

Our code is based off of code from OpenAI, who we thank. The original code and related paper from OpenAI can be found [here](https://github.com/openai/evolution-strategies-starter). The repo has been modified to run both ES and our algorithms, including our Deep Genetic Algorithm (DeepGA) locally and on AWS.

Note: The Humanoid experiment depends on [Mujoco](http://www.mujoco.org/). Please provide your own Mujoco license and binary

The article describing these papers can be found [here](https://eng.uber.com/deep-neuroevolution/)

## Visual Inspector for NeuroEvolution (VINE)
The folder `./visual_inspector` contains implementations of VINE, i.e., Visual Inspector for NeuroEvolution, an interactive data visualization tool for neuroevolution. Refer to `README.md` in that folder for further instructions on running and customizing your visualization. An article describing this visualization tool can be found [here](https://eng.uber.com/vine/).

## Accelerated Deep Neurevolution
The folder `./gpu_implementation` contains an implementation that uses GPU more efficiently. Refer to `README.md` in that folder for further instructions.

## How to run locally

clone repo

```
git clone https://github.com/uber-common/deep-neuroevolution.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install -r requirements.txt
```
If you plan to use the mujoco env, make sure to follow [mujoco-py](https://github.com/openai/mujoco-py)'s readme about how to install mujoco correctly

launch redis
```
. scripts/local_run_redis.sh
```

launch sample ES experiment
```
. scripts/local_run_exp.sh es configurations/frostbite_es.json  # For the Atari game Frostbite
. scripts/local_run_exp.sh es configurations/humanoid.json  # For the MuJoCo Humanoid-v1 environment
```

launch sample NS-ES experiment
```
. scripts/local_run_exp.sh ns-es configurations/frostbite_nses.json
. scripts/local_run_exp.sh ns-es configurations/humanoid_nses.json
```

launch sample NSR-ES experiment
```
. scripts/local_run_exp.sh nsr-es configurations/frostbite_nsres.json
. scripts/local_run_exp.sh nsr-es configurations/humanoid_nsres.json
```

launch sample GA experiment
```
. scripts/local_run_exp.sh ga configurations/frostbite_ga.json  # For the Atari game Frostbite
```

launch sample Random Search experiment
```
. scripts/local_run_exp.sh rs configurations/frostbite_ga.json  # For the Atari game Frostbite
```


visualize results by running a policy file
```
python -m scripts.viz 'FrostbiteNoFrameskip-v4' <YOUR_H5_FILE>
python -m scripts.viz 'Humanoid-v1' <YOUR_H5_FILE>
```

### extra folder
The extra folder holds the XML specification file for the  Humanoid
Locomotion with Deceptive Trap domain used in https://arxiv.org/abs/1712.06560. Use this XML file in gym to recreate the environment.
