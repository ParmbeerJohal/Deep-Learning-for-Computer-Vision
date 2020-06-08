#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_abstract.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#Below revision by Parm Johal
time ./tools/train_abstract_scene.py --cuda --rnn_cell=lstm --parallel --n_epochs=5
