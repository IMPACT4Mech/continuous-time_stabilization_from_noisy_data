[![arXiv][arxiv-shield]][arxiv-url]

# Continuous-Time Stabilization from Noisy Data

This repository contains the MATLAB code for the numerical examples of the paper:
> A. Bosso, M. Borghesi, A. Iannelli, B. Yi, G. Notarstefano, "Data-Driven Stabilization of Continuous-Time LTI Systems from Noisy Input-Output Data." 2026 European Control Conference (ECC).

Files to reproduce the numerical results of the paper:

- _scalar_example_v1.m_: numerical example of Section VI-A.
- _batch_reactor_example_v1.m_: numerical example of Section VI-B.

Auxiliary files:

- _v_noise.mat_ and _w_noise.mat_: noise signals used in _scalar_example_v1.m_.
- _LMI_test.mat_ and _rho.mat_: saved data from the simulation runs that generated Figure 2 of Section VI-B.

The code requires the installation of MOSEK and YALMIP:

MOSEK:  https://docs.mosek.com/10.2/toolbox/index.html

YALMIP: https://yalmip.github.io

## Contact

Alessandro Bosso

University of Bologna

Email: alessandro.bosso@unibo.it


[arxiv-shield]: https://img.shields.io/badge/arxiv-2511.11417-t?style=flat&logo=arxiv&logoColor=white&color=red
[arxiv-url]: https://arxiv.org/abs/2511.11417

## Acknowledgments

The research leading to these results has received funding from the European Union's Horizon Europe research and innovation program under the Marie Skłodowska-Curie Grant Agreement No. 101104404 - IMPACT4Mech. https://cordis.europa.eu/project/id/101104404
