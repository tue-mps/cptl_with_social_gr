# Continual-Learning-for-trajectory-prediction
This is a repository about the continual learning for trajectory prediction

# Requirements

The current version of the code has been tested with

- `torch 1.7.1`
- `torchvision 0.8.2`

You can run `conda install --yes --file requirements.txt` to install all dependencies.

# Running the experiments

Individual experiments can be run with `main.py`. 

You can run `python main.py --replay=generative --iters=1000 --z_dim=128 --batch_size=32 --replay_batch_size=128 --pdf --visdom`, this code will work.

Main options are:

- `--replay`: whether use generative replay model? (`none | generative`)
- `--iters`: the number of epochs. (default is `1000`)
- `--z_dim`: the dimension of hidden vary z in VAE. (default is `128`)
- `--batch_size`: the mini batch size of current dataset. (default is `32`)
- `--replay_batch_size`: the mini batch size of previous dataset. (default is `128`)
- `--lr`: the learning rate. (default is `0.001`)
- `--pdf`: whether save the results to a pdf.
- `--visdom`: on-the-fly plots during training.

# Reference

The code base borrows from [continual-learning](https://github.com/GMvandeVen/continual-learning)




