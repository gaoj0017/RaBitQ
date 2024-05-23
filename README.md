# [SIGMOD 2024] RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search

*   This is a scientific demo of RaBitQ. We will release a well-optimized library (with Python bindings) using RaBitQ very soon.

## Organization
*   The index phase of RaBitQ: `./data/rabitq.py`.
*   The query phase of RaBitQ: 
    *   `./src/ivf_rabitq.h` includes the general workflow.
    *   `./src/space.h`      includes the bitwise implementation of RaBitQ.
    *   `./src/fast_scan.h`  includes the SIMD-based implementation of RaBitQ.
*   Comments are provided in these files.


## Prerequisites
* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `./src/`.

---
## Reproduction

1. Download and preprocess the datasets. Detailed instructions can be found in `./data/README.md`.

2. Index the datasets. 
    ```sh
    ./script/index.sh
    ```
3. Test the queries of the datasets. The results are generated in `./results/`. 
    ```sh
    ./script/search.sh
    ```


