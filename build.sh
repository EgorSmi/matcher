#!/usr/bin/env python3
export ENVIRONMENT_NAME=solution-env
export ENVIRONMENT_ARCHIVE=solution-env.tar.gz

conda create -n ENVIRONMENT_NAME python=3.7

conda activate ENVIRONMENT_NAME
pip install .
conda-pack -n ENVIRONMENT_NAME -o ENVIRONMENT_ARCHIVE
