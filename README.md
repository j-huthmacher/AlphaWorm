# AlphaWorm
Student project where we use reinforcement learning to solve the worm domain of Unity ML.

## Package Structure
Some remarks about the package structure.

Within the root level of the repository you find the folder `dev` that contains all important files and also some results from the DDPG training.

**DDPG**

`dev/models`: Contain results from different training runs and different environments.

`dev/agents`: Contains one of the DDPG implementation as well as adapted copy of the TD3 implementation from `dev/td3`

`dev/notebooks`: Contains some notebooks to test the DDPG and see some plots containing the results of the DDPG.

## Usage
### Prerequisites:
Local
```
pip3 install -r dev/requirements.txt
```
### Usage:
Select DDPG/TD3 for Unity/Gym in dev/start.py main method.

Local (Set working directory to /dev/)
```
python3 dev/start.py
```

Alternatively you can directly start the `start.py` within the `dev` folder by change the directory to `dev`:

```
python start.py
```

**Important**: Depending on your execution environment you have to choose the rigth Unity build!

For example for training the DDPG you can easily comment the windows (default for DDPG) environemnt and uncomment the linux path, if you want to execute it on windows.
(start.py, lines 90 -91)

```
    env = "envs/worm_dynamic_one_agent/win/UnityEnvironment"  # Windows
    # env = "./envs/worm_dynamic_one_agent/linux/worm_dynamic"  # Linux
```

Remote (Slurm via CIP):
```
./deploy.sh
```
Run this shell file in the AlphaWorm folder if the code is stored locally.
Environment and code will get transfered to the CIP (sshpass is necessary) then the training will start automatically.
Slurm logs will be created in the home directory!

If the code is already transfered, one can just start the start.sh script to start the training.

## TD3
All hyperparameters are already set as default and that is what is working and should not be changed. This is currently the best working version.
So most of the results will refer to this algorithm.

## DDPG
You can find the DDPG implementation and the required dependent component under the following link [https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents/ddpg](https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents/ddpg). Except the `replay buffer` which is stored in the `memory_buffer.py` under [https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents](https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents)

For using the DDPG implementation please have a look on this notebook [https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-alpha-worm-ddpg.ipynb](https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-alpha-worm-ddpg.ipynb) that explains the general way how to use the implmentation.

**Results (DDPG)**

You find some of the training results as well as corresponding dumpes models in `dev/models`. Please be aware that not each training run results in an dumped model. Moreover, sometimes you will find either the file `rewards.csv` or `eval_rewards.csv`. Both files contains (more or less) the same and can be used to review the results. In case you find `rewards.csv` and `eval_rewards.csv` just use the `eval_rewards.csv` instead of `rewards.csv`.

To simplify review the results you can just have a look to the plotting notebook [https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-ddpg-plots.ipynb](https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-ddpg-plots.ipynb). 

## Errors
### Tensorflow
```
No module tensorflow.contrib
```
Install tensorflow version < 2.0 (e.g. 1.15.3)
