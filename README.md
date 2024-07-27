# TRANSFIL_python
A python code of the TRANSFIL model developed by MIke Irvine used to model the dynamics of lympahtic filariasis over time.  The model simulates individuals with varying worm burden, microfilaraemia, and demographic factors associated with age and susceptibility to infection using a stochastic micro-simulation approach.

A full model description can be description can be found here: https://parasitesandvectors.biomedcentral.com/articles/10.1186/s13071-015-1152-3.



## Model in python.
The `sim.py`file taks in the arguments 1) the number of years (in years), 2) total population and 3) runs - the number of simulations to take place. 

After simulation is done the output returned is a plot of the antigen and micro-filariae prevalence over time.

## Pros
This code can help individuals who are good at python and are not doing so well with other programming languages to understand the model dynamics

## Cons 
It takes much time to run as compared to the C++ version of it.
