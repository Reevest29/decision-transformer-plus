
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt --num_steps_per_iter 1000 --max_iters 100
```

```
python experiment.py --env hopper --dataset medium --model_type dt+wm --num_steps_per_iter 1000 --max_iters 100 --pretrain_steps_per_iter 20 --pretrain_iters 5 --a_weight 1 --r_weight 0 --s_weight 0
```

Adding `-w True` will log results to Weights and Biases.
