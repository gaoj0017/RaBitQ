#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"

using namespace std;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'd'},
        {"source",                      required_argument, 0, 's'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256]="";
    char source[256]="";
    
    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg){
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if(optarg){
                    strcpy(source, optarg);
                }
                break;
        }
    }

    
    // ==============================================================================================================
    // Load Data
    char data_path[256] = "";
    char index_path[256] = "";
    char centroid_path[256] = "";
    char x0_path[256] = "";
    char dist_to_centroid_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";

    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    Matrix<float> X(data_path);
    
    sprintf(centroid_path, "%sRandCentroid_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> C(centroid_path);

    sprintf(x0_path, "%sx0_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> x0(x0_path);

    sprintf(dist_to_centroid_path, "%s%s_dist_to_centroid_%d.fvecs", source, dataset, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);
    
    sprintf(cluster_id_path, "%s%s_cluster_id_%d.ivecs", source, dataset, numC);
    Matrix<uint32_t> cluster_id(cluster_id_path);
    
    sprintf(binary_path, "%sRandNet_C%d_B%d.Ivecs", source, numC, BB);
    Matrix<uint64_t> binary(binary_path);

    sprintf(index_path, "%sivfrabitq%d_B%d.index", source, numC, BB);
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================

    IVFRN<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);

    ivf.save(index_path);

    return 0;
}
