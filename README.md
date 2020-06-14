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
Slurm logs will be created in the home directory!

### Errors
## Tensorflow
```
No module tensorflow.contrib
```
Install tensorflow version < 2.0 (e.g. 1.15.3)
