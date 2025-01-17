# NeuralExplorer

State Space Exploration of ClosedLoop Control Systems Using Neural Networks.

## Introduction
The software first approximates the sensitivity and inverse sensitivity functions using neural network.
Then, these neural networks are used to falsify a given system or derive system simulations for reachability.
The examples in this tutorial primarily focus on inverse sensitivity; the application and usage in the context 
of forward sensitivity follows directly.

### Prerequisites and installation

* python3.0

* Jupyter notebook

* keras

* numpy

* matplotlib

* itertools

* tensorflow

* sklearn


## Running the tests

All experiments are organized in different notebooks located in eval-notebooks directory under 'eval' folder.
The user can look at these notebooks to understand their utility and run their own tests.
Although notebook titles are self explanatory, some of their descriptions are given below.

* Runs corresponding to basic algorithm 'reachTarget' using inverse sensitivity - `eval-inv-sen-basic`

<p align="center"> <img src="fig-inv-sen-basic.png" alt="reachTarget algorithm"/> </p>

* Runs where basic algorithm is performed multiple times and mean is computed - `eval-inv-sen-avg`

* Runs to perform falsification (given an unsafe set) using inverse sensitivity - `eval-inv-sen-falsification`

<p align="center"> <img src="fig-inv-sen-falsification.png" alt="inv-sen-falsification"/> </p>

* In order to generate the corresponding point cloud based on distance profiles - `eval-fwd-sen-falsify-point-cloud`

<p align="center"> <img src="fig-inv-sen-falsify-point-cloud.png" alt="inv-sen-falsify-pointcloud"/> </p>

* Runs to perform reachability using forward sensitivity - `eval-fwd-sen-reachability`

<img align="left" width="420" src="fig-fwd-sen-reach-act.png" alt="fwd-sen-reach-act">
<img align="right" width="420" src="fig-fwd-sen-reach-pred.png" alt="fwd-sen-reach-pred">

* Runs to perform point cloud while falsifying using forward sensitivity - `eval-fwd-sen-falsify-point-cloud`

<img align="left" width="400" src="fig-fwd-sen-falsify-act.png" alt="fwd-sen-falsify-act">
<img align="right" width="400" src="fig-fwd-sen-falsify-pred.png" alt="fwd-sen-falsify-pred">

## Authors

* Manish Goyal, with input from Parasara **Sridhar** Duggirala (https://www.cs.unc.edu/~psd/)


## License
The project is licensed under the GPL v3 license (see the LICENSE file).
