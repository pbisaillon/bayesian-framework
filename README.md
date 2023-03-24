# README #

Currently revamping my research code into something that's a bit more usable! This code will be unstable for a while until I clean the code and refactor some parts. I will also improve the documentation.


## Guillimin usage ##

To use on Guillimin, you will need to load the following modules

module load

Currently Loaded Modules:
  1) gcc/4.9.1   2) openmpi/1.8.3-gcc   3) GotoBLAS+LAPACK


## Requirements ##

cmake, openmpi, gcc version that supports c++11 standard, lapback and blas for amardillo

### GCC ###
The c++ compiler must support C++11. The minimum version should be 4.7 but the code has only been tested with version 5.2. If installing new GCC compilers, OpenMPI will also need to be reinstalled using those compilers.


## Installation ##

The first step is to clone the repository. You will have to be granted access first.
To clone the repository it should be something similar to
```
git clone https://pbisaillon@bitbucket.org/pbisaillon/bayesian-framework.git
```

To install the framework
```
$ cmake .
$ make
```

You will need to create a variable BFPATH that contains the path to the framework

```
$ export BFPATH=<path to repo>
```

## Usage ##

Examples of the framework are provided in the Examples folder. State estimation of the Van der Pol Oscillator is presented in Examples/VanDerPolOscillator. To compile the code necessary for this example write
```
$ cmake .
$ make
```
You can generate the data using
```
./data gcase00.cfg
```
Run the example using
```
mpirun -np <nprocs> ./run case00.cfg
```
You can use python to plot the state estimation results using
```
python draw.py
```
The results will be in the figs folder.

### Possible issues ###

If you have anaconda3 installed. Amardillo will detect hdf5 and I couldn't get it to work. I've simply renamed anaconda3 folder to a different name before the installation. After, I've renamed to the original name.

### Who do I talk to? ###

* Philippe Bisaillon (email: philippe.bisaillon@carleton.ca )
