# [SIGMOD 2024] RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search

## News and Updates

* **A library with more practical implementation techniques about RaBitQ is released at the [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library).**


*   A new blog - [Quantization in The Counterintuitive High-Dimensional Space](https://dev.to/gaoj0017/quantization-in-the-counterintuitive-high-dimensional-space-4feg) - to provide the key insights behind RaBitQ and its extension (corresponding to optimized approaches to binary quantization and scalar quantization respectively).

---

We are open to address any questions regarding the RaBitQ project. Please feel free to drop us an email at *jianyang.gao [at] ntu.edu.sg* and *c.long [at] ntu.edu.sg*.

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

---

Please cite our paper using the following bibtex if it is used in your research.

```
@article{10.1145/3654970,
author = {Gao, Jianyang and Long, Cheng},
title = {RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search},
year = {2024},
issue_date = {June 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {3},
url = {https://doi.org/10.1145/3654970},
doi = {10.1145/3654970},
abstract = {Searching for approximate nearest neighbors (ANN) in the high-dimensional Euclidean space is a pivotal problem. Recently, with the help of fast SIMD-based implementations, Product Quantization (PQ) and its variants can often efficiently and accurately estimate the distances between the vectors and have achieved great success in the in-memory ANN search. Despite their empirical success, we note that these methods do not have a theoretical error bound and are observed to fail disastrously on some real-world datasets. Motivated by this, we propose a new randomized quantization method named RaBitQ, which quantizes D-dimensional vectors into D-bit strings. RaBitQ guarantees a sharp theoretical error bound and provides good empirical accuracy at the same time. In addition, we introduce efficient implementations of RaBitQ, supporting to estimate the distances with bitwise operations or SIMD-based operations. Extensive experiments on real-world datasets confirm that (1) our method outperforms PQ and its variants in terms of accuracy-efficiency trade-off by a clear margin and (2) its empirical performance is well-aligned with our theoretical analysis.},
journal = {Proc. ACM Manag. Data},
month = may,
articleno = {167},
numpages = {27},
keywords = {Johnson-Lindenstrauss transformation, approximate nearest neighbor search, quantization}
}
```

Please provide a reference of our paper if it helps in your system.

```
Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970
```





