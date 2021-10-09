Sparse Deep Learning: A New Framework Immuneto Local Traps and Miscalibration
===============================================================

### Installation

Requirements for Pytorch see [Pytorch](http://pytorch.org/) installation instructions

### Simulation:
Generate Data:
```{python}
python Generate_Data.py
```
##### Command For Running Variable Selection Experiment
```{python}
python Simulation_Regression.py --data_index 1
```

##### Command For Running Real Data Experiment:
```{python}
python cifar_run.py --sigma0_init 1.5e-5 --sigma0_end 1.5e-6 --lambdan 1e-8 -depth 20 --seed 1
python cifar_run.py --sigma0_init 6e-5 --sigma0_end 6e-6 --lambdan 1e-9 -depth 20 --seed 1
python cifar_run.py --sigma0_init 1e-4 --sigma0_end 1e-5 --lambdan 3e-8 -depth 32 --seed 1
python cifar_run.py --sigma0_init 1e-4 --sigma0_end 1e-5 --lambdan 2e-8 -depth 32 --seed 1
```
