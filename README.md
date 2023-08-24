# PALMAD: Parallel Arbitrary Length MERLIN-based Anomaly Discovery
This repository is related to the PALMAD (Parallel Arbitrary Length MERLIN-based Anomaly Discovery) algorithm that accelerates all-length anomaly discovery in time series with a graphics processor. PALMAD is authored by Yana Kraeva (kraevaya@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PALMAD's source code (in C, CUDA), accompanying datasets, and experimental results. Please cite an article that describes PALMAD as shown below.

PALMAD is based on the MERLIN serial algorithm by Nakamura et al. [https://doi.org/10.1109/ICDM50108.2020.00147]. To discover time series anomalies, PALMAD exploits the discord concept: a time series discord is defined as a subsequence that is maximally far away from its non-overlapping nearest neighbor subsequence. As opposed to its serial predecessor, PALMAD employs recurrent formulas we have derived to avoid redundant calculations, and advanced data structures for the efficient implementation of parallel processing. 

# Citation
```
@article{ZymblerKraeva2023,
 author    = {Mikhail Zymbler and Yana Kraeva},
 title     = {High-Performance Time Series Anomaly Discovery on Graphics Processors},
 journal   = {Mathematics},
 volume    = {11},
 number    = {14},
 pages     = {3193},
 year      = {2023},
 doi       = {10.3390/math11143193},
 url       = {https://doi.org/10.3390/math11143193}
}
```
