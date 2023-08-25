import os
import shutil
from typing import Tuple

from segment_anything.finetuning.config import Config


def create_experiment(config: Config) -> Tuple[str, str]:
    """Create experiment."""
    experiment_dir = os.path.join(config.experiments_dir, config.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_path = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_path, exist_ok=True)

    # TODO copy dataset path
    # dataset_path = os.path.join(experiment_dir, 'dataset')
    # if os.path.exists(dataset_path):
    #     shutil.rmtree(dataset_path)

    return experiment_dir, checkpoints_path
