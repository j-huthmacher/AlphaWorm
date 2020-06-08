import atexit
from distutils.version import StrictVersion
import glob
import uuid
import numpy as np
import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple

import mlagents_envs

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage

from mlagents_envs.base_env import (
    BaseEnv,
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    BehaviorName,
    AgentId,
)
from mlagents_envs.timers import timed, hierarchical_timer
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityActionException,
    UnityTimeOutException,
    UnityCommunicatorStoppedException,
)

from mlagents_envs.communicator_objects.command_pb2 import STEP, RESET
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto

from mlagents_envs.communicator_objects.unity_rl_input_pb2 import UnityRLInputProto
from mlagents_envs.communicator_objects.unity_rl_output_pb2 import UnityRLOutputProto
from mlagents_envs.communicator_objects.agent_action_pb2 import AgentActionProto
from mlagents_envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from mlagents_envs.communicator_objects.capabilities_pb2 import UnityRLCapabilitiesProto
from mlagents_envs.communicator_objects.unity_rl_initialization_input_pb2 import (
    UnityRLInitializationInputProto,
)

from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto

from .rpc_communicator import RpcCommunicator
from sys import platform
import signal
import struct

logger = get_logger(__name__)


class UnityEnvironment(BaseEnv):
    SCALAR_ACTION_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)
    SINGLE_BRAIN_ACTION_TYPES = SCALAR_ACTION_TYPES + (list, np.ndarray)

    # Communication protocol version.
    # When connecting to C#, this must be compatible with Academy.k_ApiVersion.
    # We follow semantic versioning on the communication version, so existing
    # functionality will work as long the major versions match.
    # This should be changed whenever a change is made to the communication protocol.
    API_VERSION = "1.0.0"

    # Default port that the editor listens on. If an environment executable
    # isn't specified, this port will be used.
    DEFAULT_EDITOR_PORT = 5004

    # Default base port for environments. Each environment will be offset from this
    # by it's worker_id.
    BASE_ENVIRONMENT_PORT = 5005

    # Command line argument used to pass the port to the executable environment.
    PORT_COMMAND_LINE_ARG = "--mlagents-port"

    @staticmethod
    def _raise_version_exception(unity_com_ver: str) -> None:
        raise UnityEnvironmentException(
            f"The communication API version is not compatible between Unity and python. "
            f"Python API: {UnityEnvironment.API_VERSION}, Unity API: {unity_com_ver}.\n "
            f"Please find the versions that work best together from our release page.\n"
            "https://github.com/Unity-Technologies/ml-agents/releases"
        )

    @staticmethod
    def check_communication_compatibility(
        unity_com_ver: str, python_api_version: str, unity_package_version: str
    ) -> bool:
        unity_communicator_version = StrictVersion(unity_com_ver)
        api_version = StrictVersion(python_api_version)
        if unity_communicator_version.version[0] == 0:
            if (
                unity_communicator_version.version[0] != api_version.version[0]
                or unity_communicator_version.version[1] != api_version.version[1]
            ):
                # Minor beta versions differ.
                return False
        elif unity_communicator_version.version[0] != api_version.version[0]:
            # Major versions mismatch.
            return False
        elif unity_communicator_version.version[1] != api_version.version[1]:
            # Non-beta minor versions mismatch.  Log a warning but allow execution to continue.
            logger.warning(
                f"WARNING: The communication API versions between Unity and python differ at the minor version level. "
                f"Python API: {python_api_version}, Unity API: {unity_communicator_version}.\n"
                f"This means that some features may not work unless you upgrade the package with the lower version."
                f"Please find the versions that work best together from our release page.\n"
                "https://github.com/Unity-Technologies/ml-agents/releases"
            )
        else:
            logger.info(
                f"Connected to Unity environment with package version {unity_package_version} "
                f"and communication version {unity_com_ver}"
            )
        return True

    @staticmethod
    def get_capabilities_proto() -> UnityRLCapabilitiesProto:
        capabilities = UnityRLCapabilitiesProto()
        capabilities.baseRLCapabilities = True
        return capabilities

    @staticmethod
    def warn_csharp_base_capabitlities(
        caps: UnityRLCapabilitiesProto, unity_package_ver: str, python_package_ver: str
    ) -> None:
        if not caps.baseRLCapabilities:
            logger.warning(
                "WARNING: The Unity process is not running with the expected base Reinforcement Learning"
                " capabilities. Please be sure upgrade the Unity Package to a version that is compatible with this "
                "python package.\n"
                f"Python package version: {python_package_ver}, C# package version: {unity_package_ver}"
                f"Please find the versions that work best together from our release page.\n"
                "https://github.com/Unity-Technologies/ml-agents/releases"
            )

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
    ):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Unity environment binary.
        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        If no environment is specified (i.e. file_name is None), the DEFAULT_EDITOR_PORT will be used.
        :int worker_id: Offset from base_port. Used for training multiple environments simultaneously.
        :bool no_graphics: Whether to run the Unity simulator in no-graphics mode
        :int timeout_wait: Time (in seconds) to wait for connection from environment.
        :list args: Addition Unity command line arguments
        :list side_channels: Additional side channel for no-rl communication with Unity
        """
        args = args or []
        atexit.register(self._close)
        # If base port is not specified, use BASE_ENVIRONMENT_PORT if we have
        # an environment, otherwise DEFAULT_EDITOR_PORT
        if base_port is None:
            base_port = (
                self.BASE_ENVIRONMENT_PORT if file_name else self.DEFAULT_EDITOR_PORT
            )
        self.port = base_port + worker_id
        self._buffer_size = 12000
        # If true, this means the environment was successfully loaded
        self._loaded = False
        # The process that is started. If None, no process was started
        self.proc1 = None
        self.timeout_wait: int = timeout_wait
        self.communicator = self.get_communicator(worker_id, base_port, timeout_wait)
        self.worker_id = worker_id
        self.side_channels: Dict[uuid.UUID, SideChannel] = {}
        if side_channels is not None:
            for _sc in side_channels:
                if _sc.channel_id in self.side_channels:
                    raise UnityEnvironmentException(
                        "There cannot be two side channels with the same channel id {0}.".format(
                            _sc.channel_id
                        )
                    )
                self.side_channels[_sc.channel_id] = _sc

        # If the environment name is None, a new environment will not be launched
        # and the communicator will directly try to connect to an existing unity environment.
        # If the worker-id is not 0 and the environment name is None, an error is thrown
        if file_name is None and worker_id != 0:
            raise UnityEnvironmentException(
                "If the environment name is None, "
                "the worker-id must be 0 in order to connect with the Editor."
            )
        if file_name is not None:
            self.executable_launcher(file_name, no_graphics, args)
        else:
            logger.info(
                f"Listening on port {self.port}. "
                f"Start training by pressing the Play button in the Unity Editor."
            )
        self._loaded = True

        rl_init_parameters_in = UnityRLInitializationInputProto(
            seed=seed,
            communication_version=self.API_VERSION,
            package_version=mlagents_envs.__version__,
            capabilities=UnityEnvironment.get_capabilities_proto(),
        )
        try:
            aca_output = self.send_academy_parameters(rl_init_parameters_in)
            aca_params = aca_output.rl_initialization_output
        except UnityTimeOutException:
            self._close(0)
            raise

        if not UnityEnvironment.check_communication_compatibility(
            aca_params.communication_version,
            UnityEnvironment.API_VERSION,
            aca_params.package_version,
        ):
            self._close(0)
            UnityEnvironment._raise_version_exception(aca_params.communication_version)

        UnityEnvironment.warn_csharp_base_capabitlities(
            aca_params.capabilities,
            aca_params.package_version,
            UnityEnvironment.API_VERSION,
        )

        self._env_state: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}
        self._env_specs: Dict[str, BehaviorSpec] = {}
        self._env_actions: Dict[str, np.ndarray] = {}
        self._is_first_message = True
        self._update_behavior_specs(aca_output)

    @staticmethod
    def get_communicator(worker_id, base_port, timeout_wait):
        return RpcCommunicator(worker_id, base_port, timeout_wait)

    @staticmethod
    def validate_environment_path(env_path: str) -> Optional[str]:
        # Strip out executable extensions if passed
        env_path = (
            env_path.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
        true_filename = os.path.basename(os.path.normpath(env_path))
        logger.debug("The true file name is {}".format(true_filename))

        if not (glob.glob(env_path) or glob.glob(env_path + ".*")):
            return None

        cwd = os.getcwd()
        launch_string = None
        true_filename = os.path.basename(os.path.normpath(env_path))
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, env_path) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, env_path) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(env_path + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(env_path + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform == "darwin":
            candidates = glob.glob(
                os.path.join(cwd, env_path + ".app", "Contents", "MacOS", true_filename)
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(env_path + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, env_path + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(env_path + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) > 0:
                launch_string = candidates[0]
        elif platform == "win32":
            candidates = glob.glob(os.path.join(cwd, env_path + ".exe"))
            if len(candidates) == 0:
                candidates = glob.glob(env_path + ".exe")
            if len(candidates) > 0:
                launch_string = candidates[0]
        return launch_string

    def executable_launcher(self, file_name, no_graphics, args):
        launch_string = self.validate_environment_path(file_name)
        if launch_string is None:
            self._close(0)
            raise UnityEnvironmentException(
                f"Couldn't launch the {file_name} environment. Provided filename does not match any environments."
            )
        else:
            logger.debug("This is the launch string {}".format(launch_string))
            # Launch Unity environment
            subprocess_args = [launch_string]
            if no_graphics:
                subprocess_args += ["-nographics", "-batchmode"]
            subprocess_args += [UnityEnvironment.PORT_COMMAND_LINE_ARG, str(self.port)]
            subprocess_args += args
            try:
                self.proc1 = subprocess.Popen(
                    subprocess_args,
                    # start_new_session=True means that signals to the parent python process
                    # (e.g. SIGINT from keyboard interrupt) will not be sent to the new process on POSIX platforms.
                    # This is generally good since we want the environment to have a chance to shutdown,
                    # but may be undesirable in come cases; if so, we'll add a command-line toggle.
                    # Note that on Windows, the CTRL_C signal will still be sent.
                    start_new_session=True,
                )
            except PermissionError as perm:
                # This is likely due to missing read or execute permissions on file.
                raise UnityEnvironmentException(
                    f"Error when trying to launch environment - make sure "
                    f"permissions are set correctly. For example "
                    f'"chmod -R 755 {launch_string}"'
                ) from perm

    def _update_behavior_specs(self, output: UnityOutputProto) -> None:
        init_output = output.rl_initialization_output
        for brain_param in init_output.brain_parameters:
            # Each BrainParameter in the rl_initialization_output should have at least one AgentInfo
            # Get that agent, because we need some of its observations.
            agent_infos = output.rl_output.agentInfos[brain_param.brain_name]
            if agent_infos.value:
                agent = agent_infos.value[0]
                new_spec = behavior_spec_from_proto(brain_param, agent)
                self._env_specs[brain_param.brain_name] = new_spec
                logger.info(f"Connected new brain:\n{brain_param.brain_name}")

    def _update_state(self, output: UnityRLOutputProto) -> None:
        """
        Collects experience information from all external brains in environment at current step.
        """
        for brain_name in self._env_specs.keys():
            if brain_name in output.agentInfos:
                agent_info_list = output.agentInfos[brain_name].value
                self._env_state[brain_name] = steps_from_proto(
                    agent_info_list, self._env_specs[brain_name]
                )
            else:
                self._env_state[brain_name] = (
                    DecisionSteps.empty(self._env_specs[brain_name]),
                    TerminalSteps.empty(self._env_specs[brain_name]),
                )
        self._parse_side_channel_message(self.side_channels, output.side_channel)

    def reset(self) -> None:
        if self._loaded:
            outputs = self.communicator.exchange(self._generate_reset_input())
            if outputs is None:
                raise UnityCommunicatorStoppedException("Communicator has exited.")
            self._update_behavior_specs(outputs)
            rl_output = outputs.rl_output
            self._update_state(rl_output)
            self._is_first_message = False
            self._env_actions.clear()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    @timed
    def step(self) -> None:
        if self._is_first_message:
            return self.reset()
        if not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        # fill the blanks for missing actions
        for group_name in self._env_specs:
            if group_name not in self._env_actions:
                n_agents = 0
                if group_name in self._env_state:
                    n_agents = len(self._env_state[group_name][0])
                self._env_actions[group_name] = self._env_specs[
                    group_name
                ].create_empty_action(n_agents)
        step_input = self._generate_step_input(self._env_actions)
        with hierarchical_timer("communicator.exchange"):
            outputs = self.communicator.exchange(step_input)
        if outputs is None:
            raise UnityCommunicatorStoppedException("Communicator has exited.")
        self._update_behavior_specs(outputs)
        rl_output = outputs.rl_output
        self._update_state(rl_output)
        self._env_actions.clear()

    def get_behavior_names(self):
        return list(self._env_specs.keys())

    def _assert_behavior_exists(self, behavior_name: str) -> None:
        if behavior_name not in self._env_specs:
            raise UnityActionException(
                "The group {0} does not correspond to an existing agent group "
                "in the environment".format(behavior_name)
            )

    def set_actions(self, behavior_name: BehaviorName, action: np.ndarray) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        spec = self._env_specs[behavior_name]
        expected_type = np.float32 if spec.is_action_continuous() else np.int32
        expected_shape = (len(self._env_state[behavior_name][0]), spec.action_size)
        if action.shape != expected_shape:
            raise UnityActionException(
                "The behavior {0} needs an input of dimension {1} for "
                "(<number of agents>, <action size>) but received input of "
                "dimension {2}".format(behavior_name, expected_shape, action.shape)
            )
        if action.dtype != expected_type:
            action = action.astype(expected_type)
        self._env_actions[behavior_name] = action

    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: np.ndarray
    ) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        spec = self._env_specs[behavior_name]
        expected_shape = (spec.action_size,)
        if action.shape != expected_shape:
            raise UnityActionException(
                f"The Agent {0} with BehaviorName {1} needs an input of dimension "
                f"{2} but received input of dimension {3}".format(
                    agent_id, behavior_name, expected_shape, action.shape
                )
            )
        expected_type = np.float32 if spec.is_action_continuous() else np.int32
        if action.dtype != expected_type:
            action = action.astype(expected_type)

        if behavior_name not in self._env_actions:
            self._env_actions[behavior_name] = spec.create_empty_action(
                len(self._env_state[behavior_name][0])
            )
        try:
            index = np.where(self._env_state[behavior_name][0].agent_id == agent_id)[0][
                0
            ]
        except IndexError as ie:
            raise IndexError(
                "agent_id {} is did not request a decision at the previous step".format(
                    agent_id
                )
            ) from ie
        self._env_actions[behavior_name][index] = action

    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        self._assert_behavior_exists(behavior_name)
        return self._env_state[behavior_name]

    def get_behavior_spec(self, behavior_name: BehaviorName) -> BehaviorSpec:
        self._assert_behavior_exists(behavior_name)
        return self._env_specs[behavior_name]

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._close()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _close(self, timeout: Optional[int] = None) -> None:
        """
        Close the communicator and environment subprocess (if necessary).

        :int timeout: [Optional] Number of seconds to wait for the environment to shut down before
            force-killing it.  Defaults to `self.timeout_wait`.
        """
        if timeout is None:
            timeout = self.timeout_wait
        self._loaded = False
        self.communicator.close()
        if self.proc1 is not None:
            # Wait a bit for the process to shutdown, but kill it if it takes too long
            try:
                self.proc1.wait(timeout=timeout)
                signal_name = self.returncode_to_signal_name(self.proc1.returncode)
                signal_name = f" ({signal_name})" if signal_name else ""
                return_info = f"Environment shut down with return code {self.proc1.returncode}{signal_name}."
                logger.info(return_info)
            except subprocess.TimeoutExpired:
                logger.info("Environment timed out shutting down. Killing...")
                self.proc1.kill()
            # Set to None so we don't try to close multiple times.
            self.proc1 = None

    @classmethod
    def _flatten(cls, arr: Any) -> List[float]:
        """
        Converts arrays to list.
        :param arr: numpy vector.
        :return: flattened list.
        """
        if isinstance(arr, cls.SCALAR_ACTION_TYPES):
            arr = [float(arr)]
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if len(arr) == 0:
            return arr
        if isinstance(arr[0], np.ndarray):
            # pylint: disable=no-member
            arr = [item for sublist in arr for item in sublist.tolist()]
        if isinstance(arr[0], list):
            # pylint: disable=not-an-iterable
            arr = [item for sublist in arr for item in sublist]
        arr = [float(x) for x in arr]
        return arr

    @staticmethod
    def _parse_side_channel_message(
        side_channels: Dict[uuid.UUID, SideChannel], data: bytes
    ) -> None:
        offset = 0
        while offset < len(data):
            try:
                channel_id = uuid.UUID(bytes_le=bytes(data[offset : offset + 16]))
                offset += 16
                message_len, = struct.unpack_from("<i", data, offset)
                offset = offset + 4
                message_data = data[offset : offset + message_len]
                offset = offset + message_len
            except Exception:
                raise UnityEnvironmentException(
                    "There was a problem reading a message in a SideChannel. "
                    "Please make sure the version of MLAgents in Unity is "
                    "compatible with the Python version."
                )
            if len(message_data) != message_len:
                raise UnityEnvironmentException(
                    "The message received by the side channel {0} was "
                    "unexpectedly short. Make sure your Unity Environment "
                    "sending side channel data properly.".format(channel_id)
                )
            if channel_id in side_channels:
                incoming_message = IncomingMessage(message_data)
                side_channels[channel_id].on_message_received(incoming_message)
            else:
                logger.warning(
                    "Unknown side channel data received. Channel type "
                    ": {0}.".format(channel_id)
                )

    @staticmethod
    def _generate_side_channel_data(
        side_channels: Dict[uuid.UUID, SideChannel]
    ) -> bytearray:
        result = bytearray()
        for channel_id, channel in side_channels.items():
            for message in channel.message_queue:
                result += channel_id.bytes_le
                result += struct.pack("<i", len(message))
                result += message
            channel.message_queue = []
        return result

    @timed
    def _generate_step_input(
        self, vector_action: Dict[str, np.ndarray]
    ) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        for b in vector_action:
            n_agents = len(self._env_state[b][0])
            if n_agents == 0:
                continue
            for i in range(n_agents):
                action = AgentActionProto(vector_actions=vector_action[b][i])
                rl_in.agent_actions[b].value.extend([action])
                rl_in.command = STEP
        rl_in.side_channel = bytes(self._generate_side_channel_data(self.side_channels))
        return self.wrap_unity_input(rl_in)

    def _generate_reset_input(self) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        rl_in.command = RESET
        rl_in.side_channel = bytes(self._generate_side_channel_data(self.side_channels))
        return self.wrap_unity_input(rl_in)

    def send_academy_parameters(
        self, init_parameters: UnityRLInitializationInputProto
    ) -> UnityOutputProto:
        inputs = UnityInputProto()
        inputs.rl_initialization_input.CopyFrom(init_parameters)
        return self.communicator.initialize(inputs)

    @staticmethod
    def wrap_unity_input(rl_input: UnityRLInputProto) -> UnityInputProto:
        result = UnityInputProto()
        result.rl_input.CopyFrom(rl_input)
        return result

    @staticmethod
    def returncode_to_signal_name(returncode: int) -> Optional[str]:
        """
        Try to convert return codes into their corresponding signal name.
        E.g. returncode_to_signal_name(-2) -> "SIGINT"
        """
        try:
            # A negative value -N indicates that the child was terminated by signal N (POSIX only).
            s = signal.Signals(-returncode)  # pylint: disable=no-member
            return s.name
        except Exception:
            # Should generally be a ValueError, but catch everything just in case.
            return None
