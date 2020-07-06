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

### Errors
## Tensorflow
```
No module tensorflow.contrib
```
Install tensorflow version < 2.0 (e.g. 1.15.3)
