# Performance Prediction Toolkit (PPT)
Predict the performance of physics codes

## Authors
1. Gopinath (Nath) Chennupati (gchennupati@lanl.gov)
2. Nanadakishore Santhi (nsanthi@lanl.gov)
3. Stephen Eidenbenz (eidenben@lanl.gov)
4. Robert Joseph (Joe) Zerr (rzerr@lanl.gov)
5. Massimiliano (Max) Rosa (maxrosa@lanl.gov)
6. Richard James Zamora (rjzamora@lanl.gov)
7. Eun Jung (EJ) Park (ejpark@lanl.gov)
8. Balasubramanya T (Balu) Nadiga (balu@lanl.gov)
###### Non-LANL Authors
9. Jason Liu (luix@cis.fiu.edu)
10. Kishwar Ahmed
11. Mohammad Abu Obaida
12. Yehia Arafa (yarafa@nmsu.edu)

## Recent News 

We are releasing a new version of PPT, called as [ppt_lite](./ppt_lite).

## Installation

###### Dependencies
PPT depends on another LANL licensed open source software package named SIMIAN PDES located at https://github.com/pujyam/simian.

**Simian** relies on python package _greenlet_
_pip install greenlet_

PPT installation is simple, just checkout the code as follows
> git clone https://github.com/lanl/PPT.git

## Usage
Let's assume the PPT is cloned into your home directory (/home/user/PPT)
In the _code_ directory PPT is organized in three main layers:

1. **hardware** -- contains the models for various CPUs and GPUs, interconnects,
2. **middleware** -- contains the models for MPI, OpenMP, etc.
3. **apps** -- contains various examples. These examples are the stylized pseudo (mini) apps for various open sourced physics codes.

### Runnning PPT in _Serial mode_

For example, we run SNAP simulator in serial with one of the PPT hardware models as follows:

> cd ~/PPT/code/apps/snapsim

> python snapsim-orig.py in >> out

where, _in_ and _out_ in the above command are input and output files of SNAPsim.

## Classification
PPT is Unclassified and contains no Unclassified Controlled Nuclear Information. It abides with the following computer code from Los Alamos National Laboratory
* Code Name: Performance Prediction Toolkit, C17098
* Export Control Review Information: DOC-U.S. Department of Commerce, EAR99
* B&R Code: YN0100000

## License
&copy 2017. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
 
All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
 
Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions.
Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file:
This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
