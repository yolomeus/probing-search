import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from procedures import DefaultTraining


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file, or continue from a checkpoint"""

    seed_everything(cfg.random_seed, workers=True)

    training = DefaultTraining(cfg)
    training.run()


if __name__ == '__main__':
    train()
