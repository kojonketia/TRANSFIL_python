# TRANSFIL_python
A python code of the TRANSFIL model developed by MIke Irvine used to model the dynamics of lympahtic filariasis (LF) over time.  The model simulates individuals with varying worm burden, microfilaraemia, and demographic factors associated with age and susceptibility to infection using a stochastic micro-simulation approach.

A full model description can be description can be found here: https://parasitesandvectors.biomedcentral.com/articles/10.1186/s13071-015-1152-3.

## Model in python.

The **simulation with plots** folder contains the python file of the model, the script file to run the model and the parameters in a txt file.

The `sim.py` file taks in the arguments 1) the number of years (in years), 2) total population and 3) runs - the number of simulations to take place. 
The `parameter.txt` file contains the required parameters needed to run the model. This can be modified depending on the setting considered to study the dynamics of LF.

Simulation is intiated in the terminal with the command below. First suppose the number of years is 120, total population is 1000 and running the model 50 times then we have

```terminal
python sim.py 120 1000 50
```

After simulation is done the output returned is a plot of the antigen and micro-filariae prevalence over time.

## Pros
This code can help individuals who are good at python and are not doing so well with other programming languages to understand the model dynamics

## Cons 
It takes much time to run as compared to the C++ version of it.
