# A Parallel Algorithm for Continuous-Time Constrained Model Predictive Control

This repository is the code for linear MPC using Parallelized Primal-Dual Hybrid Gradient Algorithm. This is the paper (conference version)[link](https://arxiv.org/abs/2303.17889).


In this paper, we consider a Model Predictive Control (MPC) problem of a continuous-time linear time-invariant system subject to continuous-time path constraints on the states and the inputs. By leveraging the concept of differential flatness, we can replace the differential equations governing the system with linear mapping between the states, inputs, and flat outputs (including their derivatives). The flat outputs are then parameterized by piecewise polynomials, and the model predictive control problem can be equivalently transformed into a Semi-Definite Programming (SDP) problem via Sum-of-Squares (SOS), ensuring constraint satisfaction at every continuous-time interval. We further note that the SDP problem contains a large number of small-size semi-definite matrices as optimization variables. To address this, we develop a Primal-Dual Hybrid Gradient (PDHG) algorithm that can be efficiently parallelized to speed up the optimization procedure. Simulation results on a quadruple-tank process demonstrate that our formulation can guarantee strict constraint satisfaction, while the standard MPC controller based on the discretized system may violate the constraint inside a sampling period. Moreover, the computational speed superiority of our proposed algorithm is collaborated by the simulation.


Our proposed methods provides fast computation with continuous-time constraint satisfaction guarantee. The main procedure is solving an SDP problem with customized primal-dual based iterations in a parallelized manner.

The project is written in [julia](https://julialang.org/).

The main test script is **quadruple_example.jl**. Please run this script for numerical simulation.

## Result

![Visualiztion of the optimziation process](https://github.com/zs-li/MPC_PDHG/blob/main/anim.gif)

This figure shows the optimization intermediate states of our proposed algorithm. As optimization goes on, the polynomials representing the state trajectories are converging to the actual trajectories governed by system dynamics and subject to constraints. 

"k" in the title is the number of apply steps, also the index of sequential SDP problems we solve.

The shift warm start strategy can speed up the optimization process by setting the initial values in the overlapped horizon as in the previous step. Only the final segment (**new** horizon) is needed to be calculated from a random intial value.

## package requirements

The julia version we use is:

```
julia v1.9.4
```

The package we use for code test are:

```
NumericalIntegration v0.3.3
TensorOperations v3.2.4
CUDA v3.13.1
```

Auxiliary code for visualization are:

```
DynamicPolynomials v0.4.6
Plots v1.38.8
```

For compatibility constraints, ```TensorOperations``` requires that ```CUDA``` version \< v4.1.4.

## Julia environment setup

```
julia> ]
(@v1.9) pkg> activate .
(@v1.9) pkg> instantiate
```
