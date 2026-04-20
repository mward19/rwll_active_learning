# BYU Supercomputing Environment Setup

The Graphlearning package relies on the Annoy (Approximate Nearest Neighbors Oh Yeah) package to generate graph data. The Annoy package is a C++ library with Python bindings. Due to its underlying C++ nature, Annoy is compiled according to the instructions of where it is installed. For the BYU Supercomputer, this presents a problem since the login nodes have a newer instruction set than the compute nodes. As such, we present the following janky way to get the environment running on the compute node.

## Instructions

1. Build as much of the environment on the login node as possible via:

> mamba env create -f environment.yml

> eval "$(mamba shell hook --shell bash)" // This line may be optional if your terminal already recognizes mamba

> mamba activate rwll_env

2. The compute nodes don't have access to the interwebs. Thus, heretofore, henceforth, etc. etc. we will need to download the packages (not install) to a local directory to be installed on the compute nodes momentarily.
3. We use cuda version 12.6 since this currently (as of 4/20/26) works with the P100s on the BYU Supercomputer. Different cuda versions will be required for the higher end GPUs (H100s, A100s, etc.), but many of those other GPUs also require the job to be preemptable. See https://rc.byu.edu/wiki/?id=Getting+Started+With+GPUs for more information.

> mkdir wheels

> python -m pip download --index -url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio -d wheels

> python -m pip download annoy graphlearning nose -d wheels

4. From here we now begin an interactive session with a compute node (again, if you are working with newer GPUs you will need to add a constraint to ensure you get onto a node that has that GPU)

> srun --mem 1GB --time 00:20:00 --pty bash

5. Assumming nothing has majorly imploded during the process, you should now see that your bash terminal has gone from being a mere login node to now having all the power of the sun in the palm of your hand (aka you are on a compute node). From here we can now install the rest of the packages from the wheels directory.

> python -m pip install --no-index --find-links=./wheels --no-binary=annoy --no-build-isolation torch torchvision torchaudio nose annoy graphlearning

6. The environment is now built and can run with supercompute jobs. It only needs to be built this once and will persist across job runs.