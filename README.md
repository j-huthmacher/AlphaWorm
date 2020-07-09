# AlphaWorm
Student project where we use reinforcement learning to solve the worm domain of Unity ML.

## Usage
### Prerequisites:
Local
```
pip3 install -r dev/requirements.txt
```
### Usage:
Local (Set working directory to /dev/)
```
python3 dev/main.py
```
Remote (Slurm via CIP):
```
./deploy.sh
```
Run this shell file in the AlphaWorm folder if the code is stored locally.
Environment and code will get transfered to the CIP (sshpass is necessary) then the training will start automatically.
Slurm logs will be created in the home directory!

If the code is already transfered, one can just start the start.sh script to start the training.

#### TD3
All hyperparameters are already set as default and that is what is working and should not be changed.

#### DDPG
You can find the DDPG implementation and the required dependent component under the following link [https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents/ddpg](https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents/ddpg). Except the `replay buffer` which is stored in the `memory_buffer.py` under [https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents](https://github.com/j-huthmacher/AlphaWorm/tree/master/dev/agents)

For using the DDPG implementation please have a look on this notebook [https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-alpha-worm-ddpg.ipynb](https://github.com/j-huthmacher/AlphaWorm/blob/master/dev/notebooks/jh-alpha-worm-ddpg.ipynb) that explains the general way how to use the implmentation.

### Errors
## Tensorflow
```
No module tensorflow.contrib
```
Install tensorflow version < 2.0 (e.g. 1.15.3)
