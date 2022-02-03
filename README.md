# Modeling and Interpolation of the Ambient Magnetic Field by Gaussian Processes: Python Implementation

Python implementation of the online learning algorithm proposed by [Solin et al.](https://arxiv.org/pdf/1509.04634.pdf) in order to apply it on the EMT system developed by [Biomedical Design Laboratory](https://biodesignucc.ie/build/html/index.html) ([University College Cork](https://www.ucc.ie/en/), Ireland). 

## Core idea
The basic idea of the paper is that the magnetic field can be modeled through a Gaussian Process, and an [approximation](https://arxiv.org/pdf/1401.5508.pdf) through an eigenfunction expansion of the Laplace operator in a cuboid volume helps to mitigate the computational complexity of the method.
