import logging
import os
import sys
import yaml
import datetime
import random
from copy import deepcopy
from pathlib import Path

import numpy
import torch
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
# from sacred.utils import apply_backspaces_and_linefeeds

try:
    from collections import Mapping    # until python 3.10
except ImportError:
    from collections.abc import Mapping    # from python 3.10

from run import run
from utils.custom_logging import PyMARLLogger

from utils.torch_optimizer import optimize_tensor_display


# Optimize tensor display during debugging. This is only useful for debugging.
optimize_tensor_display(torch)

current_file_path = Path(__file__).resolve()
work_dir = current_file_path.parents[1]
results_path = work_dir / "results"

ex = Experiment("pymarl", save_git_info=False)
# set to "no" if you want to see stdout/stderr in console "sys" / "no" / "fd"
SETTINGS["CAPTURE_MODE"] = "no"
# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    
    # Set the seed for all randomness sources
    # Python seed
    random.seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    # Numpy seed
    numpy.random.seed(config["seed"])
    # Torch seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # SMAC seed.
    config["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == "__main__":
    # region read_parameters
    # Params order is command line -> algs -> envs -> defaults
    params = deepcopy(sys.argv)
    # th.set_num_threads(1)
    # params = ["src/main.py", "--config=qmix", "--env-config=meltingpot", "with", "env_args.substrate_name='clean_up'"]  # Shit mountain.

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # endregion

    # Generate unique token.
    if "map_name" in config_dict["env_args"]:
        map_name = config_dict["env_args"]["map_name"]
    elif "key" in config_dict["env_args"]:
        map_name = config_dict["env_args"]["key"]
    elif "substrate_name" in config_dict["env_args"]:
        map_name = config_dict["env_args"]["substrate_name"]

    # Get experiment name from yaml
    experiment_name = config_dict["name"]

    # Update map name and experiment name from command line parameters.
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]
        elif param.startswith("name"):
            experiment_name = param.split("=")[1]

    unique_token = (
        f"{experiment_name}__{map_name}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    config_dict.update({"unique_token": unique_token})

    logger = PyMARLLogger(
        name="main",
        level=logging.DEBUG,
        file_log=True,
        log_dir=results_path / "logs",
        log_file_name=unique_token + ".log",
    )
    ex.logger = logger.logger

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in \"./results/sacred\".")
    file_obs_path = results_path / f"sacred/{unique_token}"

    ex.observers.append(
        FileStorageObserver(file_obs_path, copy_artifacts=False, copy_sources=False)
    )
    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    # ex.observers.append(MongoObserver())

    # Run experiment.
    ex.add_config(config_dict)
    ex.run_commandline(params)
