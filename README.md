# Robust-Multi-Task-Learning# Optimization for Machine Learning Project

This repository contains the implementation of the experimental section for the Robust Multi-Task Learning project.

## Structure
* `Code` - Folder containing the implementation.
  * `DatasetGenerator.py` - Contains the generation of the dataset.
  * `Utils.py` - Contains utilities functions.
  * `Plots.py` - Contains implementation of the function to plot the results.
  * `Optimizer.py` - Contains the implementation of the Proximal Algorithm and FWT Algorithm.
  * `Optimizer_AMHT_LRS.py` - Contains the implementation of the AMHT_LRS algorithm.
  * `run_prox.py` - Contains an example of how to optimize with proximal gradient descent.
  * `run_FWT.py` - Contains an example of how to optimize with FWT.
  * `run_AMHT_LRS.py` - Contains an example of how to optimize AMTH_LRS algorithm.
* `Results` - Folder containing the results of the experiments.
  * `plot-1-2` - Folder containing the collected results for the first and second plot.
  * `plot-3` - Folder containing the collected results for the third plot.
* `report.pdf` - Report pdf file.
* `requirements.txt` - Requirements text file.

## Installation
To clone the following repository, please run:\
`git clone --recursive https://https://github.com/lucarossi9/Robust-Multi-Task-Learning.git`

## Requirements
Requirements for the needed packages are available in requirements.txt. To install the needed packages, please run:\
`pip install -r requirements.txt`

## Report
The report in pdf format can be found in under the name report.pdf.

## Acknowledgements
Thanks to the TML for the amazing supervision during the project.

