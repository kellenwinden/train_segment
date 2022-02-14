#!/bin/bash

#SBATCH --partition=bch-compute             # queue to be used
#SBATCH --time=20:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --ntasks=16              # Number of cpu cores required
#SBATCH --mem=16GB               # Amount of RAM memory needed

source /programs/biogrids.shrc
export PYTHON_X=3.8.8
source venv/bin/activate
python main.py
