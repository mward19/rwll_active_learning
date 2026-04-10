# one-time recommendation
conda config --set channel_priority strict

# only needed if you are solving on a login node that does not expose a GPU
# BYU RC Login node does have a GPU that can't be used for computation, but can be used to build the environment
# export CONDA_OVERRIDE_CUDA=12.8

# create new env
mamba env create -f environment.yml

# OR update an existing env
# mamba env update -n rwll_env -f environment.yml --prune

eval "$(mamba shell hook --shell bash)"
mamba activate rwll_env

# install the torch stack from the official CUDA 12.6 wheel index for the P100s
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch torchvision torchaudio

# install graphlearning and annoy last, without letting pip replace conda-managed deps
python -m pip install --no-deps graphlearning annoy