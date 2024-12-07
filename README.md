# [SIGMOD 2024] RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search

---
## News and Updates
*   Based on the questions we have received, we would like to provide the following updates and clarification regarding the RaBitQ project.
*   Q1: Does RaBitQ supports compression rates other than 32x?
*   A1: Yes. Around 2-3 months ago (Oct-2024), we released a new version of RaBitQ at https://github.com/VectorDB-NTU/Extended-RaBitQ. Based on the idea of RaBitQ, it further supports to quantize a vector with any arbitrary compression rates with asymptotically optimal accuracy. Based on our experiments, this algorithm brings especially significant improvement in the setting from 2-bit to 6-bit, which helps an algorithm to achieve high recall without reranking. 
    *   Intuitively, the algorithm achieves the promising performance because it "constructs" the codebook for every vector individually. Specifically, our method first considers a codebook formed by $B$-bit uint vectors; and then transform the codebook, i.e., translating and rescaling, on a per-vector basis. By trying different parameters of transforming, the algorithm can find a vector in the transformed codebook, that best approximate a data vector.
    *   Geometrically, the operation is equivalent to (1) normalizing both data vectors and the vectors in codebooks onto high-dim unit spheres and; (2)find every data vector's best match in the codebook after the normalization operation.
    *   Theoretically, we prove that based on the technology above, our algorithm provides an asymptotically optimal error bound regarding the trade-off between the error and the space consumption. To our knowledge, our algorithm is also the first practical algorithm that achieves the optimality. 

*   Q2: Can RaBitQ be combined with graphs? 
*   A2: Yes. We released the code at https://github.com/VectorDB-NTU/SymphonyQG. In this project, we follow the framework proposed by NGT-QG (https://github.com/yahoojapan/NGT) and combines RaBitQ and graph-based indices (NSG) in a more symphonious way. This repo is heavily optimized and achieves 3.5x-17x acceleration at 95% recall compared with the classical HNSWlib across datasets. More details about the project can be found in the paper https://arxiv.org/abs/2411.12229 that has been recently accepted by SIGMOD'25. 

*   Q3: Will Python interfaces be provided? 
*   A3: We planned to provide a simple Python interface for the basic version of RaBitQ, but later the projects related to RaBitQ become larger than our initial expectation. Currently, only https://github.com/VectorDB-NTU/SymphonyQG provides a Python interface. We are working towards a library that integrates the features regarding RaBitQ. In the library, we shall provide Python bindings. More features related to RaBitQ is upcoming.

We are open to address any questions regarding the RaBitQ project. Please feel free to drop us an email at *jianyang.gao [at] ntu.edu.sg* and *c.long [at] ntu.edu.sg*.

---
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
'''
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
'''

Please provide a reference of our paper if it helps in your system.
'''
Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970
'''





