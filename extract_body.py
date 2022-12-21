import os
from pathlib import Path

import torch
from tqdm import tqdm


def get_ckpt_paths(base_dir):
    ckpt_paths = []
    for root, dirs, files in os.walk(base_dir):
        if root.endswith('checkpoints'):
            # assert len(files) == 1
            ckpt_paths.append(os.path.join(root, files[0]))

    return ckpt_paths


def main():
    ckpt_paths = get_ckpt_paths('data/mtl_checkpoints')

    for ckpt_path in tqdm(ckpt_paths, total=len(ckpt_paths)):
        state_dict = torch.load(ckpt_path)['state_dict']
        new_state_dict = {k.replace('model.body.bert.', ''): v
                          for k, v in state_dict.items()}

        base_path = Path(os.path.split(ckpt_path)[0])
        new_path = base_path / 'state_dict.pt'

        torch.save(new_state_dict, new_path)


if __name__ == '__main__':
    main()
