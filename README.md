# BayeStab

This repo contains the code for our paper " BayeStab : Predicting effects of mutations on protein sta-bility with uncertainty quantification"

by Shuyu Wang*, Hongzhou Tang

Here we pioneered a deep graph neural network based method for predicting protein stability change upon mutation. 

![image](https://github.com/shuyu-wang/ProS-GNN/raw/main/fig1(A).png)

# Dependency

* Python 3.7
* Pytorch
* RDKit
* Rosetta
* numpy
* CUDA
* sklearn

# Installation

ProS-GNN ΔΔG  prediction is accomplished through a multi-step protocol. Each step relies on a specific third-party software that needs to be installed first. In the following, we outline the steps to install them.

### Clone BayeStab

Clone BayeStab to a local directory.

```
git clone https://github.com/HongzhouTang/BayeStab.git
```

### Install Rosetta 3

1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.13 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.




