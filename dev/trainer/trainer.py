"""
    Abstract trainer class for a unified interface.

    @author: jhuthmacher
"""

from optuna import Trial, Study


class Trainer(object):
    """ Abstract trainer object for bundle trainings procedures
    """

    def __init_(self):
        """ Initilization
        """

    def start_training(env: any, trials: int = 1, render: bool = False,
                       name: str = None):
        """ Function to exectue the individual training function.

            Parameters:
            -----------
                env: UnityEnvironment or GymEnvironment
                    The environment the agent interacts in. Could be a Unity
                    environment wrapped in a gym environment or directly a
                    gym environment.
                trials: int
                    Number of training runs that are executed from the HPO
                    library (trials = 2 means the train method is two times
                    executed with different parameters)
                render: bool
                    Flag to decide if we want to render in case of a gym
                    environment.
                name: str
                    A name for the model/agent that is used to store the best
                    model.
        """
        raise NotImplementedError

    def train(trial: Trial, env: any, render: bool = False, name: str = None,
              study: Study = None):
        """ Individual train function that implement the actual training procedure.

            Parameters:
            -----------
                trial: optuna.Trial
                    The trail object from the handovered from the HPO library.
                    This contains functionality for selecting hyper parameters.
                env: UnityEnvironment or GymEnvironment
                    The environemnt in which the agent should train.
                render: bool
                    Falg to decide if we want to render the steps in case of a
                    gym environment.
                name: str
                    A name for the model/agent that is used to store the best
                    model.
                study: optuna.Study
                    Study object that contains informationen about the training
                    (parameters for each run, best parameter set etc.)
            Return:
            -------
                reward: float
                    Since we want to optimize the reward in our case we return
                    the reward of the trainings run.
        """
        raise NotImplementedError
