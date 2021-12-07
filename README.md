# Probing Search

## Requirements

The code is tested using python 3.9 in a virtual anaconda environment. All required packages can be found
in `environment.yml` and installed using [anaconda](anaconda.com/products/individual)
like so:

```shell
conda env create -f environment.yml
```

The packages listed can also be installed manually using pip. (Note: cudatoolkit is optional and only needed for gpu
training. Same goes for wandb if you do not wish to use wandb for logging)

## Experiment Configuration

This project uses [hydra](https://github.com/facebookresearch/hydra) for composing a training configuration from
multiple sub-configuration modules. All configuration files can be found in `conf/` and its subdirectories, where
`config.yaml` is the main configuration file and subdirectories contain possible sub-configurations.

Even though yaml files define the default configuration and its structure, all training parameters can be overriden via
command line arguments when running the `train.py` script and sub-configurations can be replaced allowing easy plug and
play of training components.

### Examples

Replace the `datamodule`'s default `dataset` configuration with the `ontonotes_ner` configuration -> use `ontonotes_ner`
as dataset:

```shell
python train.py datamodule/dataset=ontonotes_ner
```

Sweep over learning rates (multirun):

```shell
python -m train.py loop.optimizer.lr=0.01,0.1,0.2
```

Do a grid search over all learning rate and batch size combinations:

```shell
python -m train.py loop.optimizer.lr=0.01,0.1 training.batch_size=32,64,128
```

## Project Structure

The configuration structure mirrors the project's structure:

```
conf/
├── datamodule
│   ├── dataset
            # dataset's task type  
│   │   └── task
        # applied before passing to model
│   └── preprocessor
|
    # defines the optimization process: loop, loss, optimizer 
├── loop (pl.LightningModule)
|
    # model to be trained in loop
└ ── model (torch.nn.Module)
    # submodules for the probing-pair model
    ├── pooler
    ├── probing_model
    └── subject_model
```

The training is instantiated in `train.py`.