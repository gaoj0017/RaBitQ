// ==================================================================
// IVFRN() involves pre-processing steps (e.g., packing the
// quantization codes into a batch) in the index phase.

// search() is the main function of the query phase.
// ==================================================================
#pragma once
#define RANDOM_QUERY_QUANTIZATION
#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include "matrix.h"
#include "utils.h"
#include "space.h"
#include "fast_scan.h"

template <uint32_t D, uint32_t B>
class IVFRN{
private:
public:
    struct Factor{
        float sqr_x;
        float error;
        float factor_ppc;
        float factor_ip;
    };

    Factor * fac;
    static constexpr float fac_norm = const_sqrt(1.0 * B);
    static constexpr float max_x1 = 1.9 / const_sqrt(1.0 * B-1.0);
    
    static Space<D,B> space;

    uint32_t N;                       // the number of data vectors 
    uint32_t C;                       // the number of clusters

    uint32_t* start;                  // the start point of a cluster
    uint32_t* packed_start;           // the start point of a cluster (packed with batch of 32)
    uint32_t* len;                    // the length of a cluster
    uint32_t* id;                     // N of size_t the ids of the objects in a cluster
    float * dist_to_c;                // N of floats distance to the centroids (not the squared distance)
    float * u;                        // B of floats random numbers sampled from the uniform distribution [0,1]

    uint64_t * binary_code;           // (B / 64) * N of 64-bit uint64_t
    uint8_t * packed_code;            // packed code with the batch size of 32 vectors

    
    float * x0;                       // N of floats in the Random Net algorithm
    float * centroid;                 // N * B floats (not N * D), note that the centroids should be randomized
    float * data;                     // N * D floats, note that the datas are not randomized    

    IVFRN();
    IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid,
            const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary);
    ~IVFRN();

    ResultHeap search(float* query, float* rd_query, uint32_t k, uint32_t nprobe, float distK = std::numeric_limits<float>::max()) const;
    
    static void scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint64_t *quant_query, uint64_t *ptr_binary_code,  uint32_t len, Factor *ptr_fac, \
                        const float  sqr_y, const float vl, const float  width, const float sumq,\
                        float *query, float *data, uint32_t *id);

    static void fast_scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code, uint32_t len, Factor *ptr_fac, \
                        const float  sqr_y, const float vl, const float  width, const float sumq,\
                        float *query, float *data, uint32_t *id);

    void save(char* filename);
    void load(char* filename);
};

// scan impl
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint64_t *quant_query, uint64_t *ptr_binary_code,  uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id){
                            
    constexpr int SIZE = 32;
    float y = std::sqrt(sqr_y);
    float res[SIZE];
    float *ptr_res = &res[0];
    int it = len / SIZE;

    for(int i=0;i<it;i++){   
        ptr_res = &res[0];
        for(int j=0;j<SIZE;j++){
            float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 -sumq) * (ptr_fac -> factor_ip) * width;
            float error_bound = y * (ptr_fac -> error);
            *ptr_res = tmp_dist - error_bound;
            ptr_binary_code += B / 64;
            ptr_fac ++;
            ptr_res ++;
        }
        
        ptr_res = &res[0];
        for(int j=0;j<SIZE;j++){
            if(*ptr_res < distK){
                
                float gt_dist = sqr_dist<D>(query, data);
                if(gt_dist < distK){
                    KNNs.emplace(gt_dist, *id);
                    if(KNNs.size() > k) KNNs.pop();
                    if(KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += D;
            ptr_res++;
            id++;
        }
    }
    
    ptr_res = &res[0];
    for(int i=it * SIZE;i<len;i++){
        float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 -sumq) * (ptr_fac -> factor_ip) * width;
        float error_bound = y * (ptr_fac -> error);
        *ptr_res = tmp_dist - error_bound;
        ptr_binary_code += B / 64;
        ptr_fac ++;
        ptr_res ++;
    }
    
    ptr_res = &res[0];
    for(int i=it * SIZE;i<len;i++){
        if(*ptr_res < distK){
            float gt_dist = sqr_dist<D>(query, data);
            if(gt_dist < distK){
                KNNs.emplace(gt_dist, *id);
                if(KNNs.size() > k) KNNs.pop();
                if(KNNs.size() == k)distK = KNNs.top().first;
            }
        }
        data += D;
        ptr_res++;
        id++;
    }
}

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::fast_scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code,  uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id){
    
    for(int i=0;i<B/4*16;i++)LUT[i] *= 2;

    float y = std::sqrt(sqr_y);

    constexpr uint32_t SIZE = 32;
    uint32_t it = len / SIZE;
    uint32_t remain = len - it * SIZE;
    uint32_t nblk_remain = (remain + 31) / 32;
    
    while(it --){
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate<B>((SIZE / 32), packed_code, LUT, result);
        packed_code += SIZE * B / 8;

        for(int i=0;i<SIZE;i++){
            float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (result[i]-sumq) * (ptr_fac -> factor_ip) * width;
            float error_bound = y * (ptr_fac -> error);
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac ++;
            ptr_low_dist ++;
        }
        ptr_low_dist = &low_dist[0];
        for(int j=0;j<SIZE;j++){
            if(*ptr_low_dist < distK){
                
                float gt_dist = sqr_dist<D>(query, data);
                // cerr << *ptr_low_dist << " " << gt_dist << endl;
                if(gt_dist < distK){
                    KNNs.emplace(gt_dist, *id);
                    if(KNNs.size() > k) KNNs.pop();
                    if(KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += D;
            ptr_low_dist++;
            id++;
        }
    }
    
    {
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate<B>(nblk_remain, packed_code, LUT, result);

        for(int i=0;i<remain;i++){
            float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (result[i] - sumq) * ptr_fac -> factor_ip * width;
            float error_bound = y * (ptr_fac -> error);
            
            // ***********************************************************************************************
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac ++;
            ptr_low_dist ++;
        }
        ptr_low_dist = &low_dist[0];
        for(int i=0;i<remain;i++){
            if(*ptr_low_dist < distK){
                
                float gt_dist = sqr_dist<D>(query, data);
                if(gt_dist < distK){
                    KNNs.emplace(gt_dist, *id);
                    if(KNNs.size() > k) KNNs.pop();
                    if(KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += D;
            ptr_low_dist++;
            id++;
        }
    }
}

// search impl
template <uint32_t D, uint32_t B>
ResultHeap IVFRN<D, B>::search(float* query, float* rd_query, uint32_t k, uint32_t nprobe, float distK) const{
    // The default value of distK is +inf 
    ResultHeap KNNs;
    // ===========================================================================================================
    // Find out the nearest N_{probe} centroids to the query vector.
    Result centroid_dist[numC];
    float * ptr_c = centroid;
    for(int i=0;i<C;i++){
        centroid_dist[i].first = sqr_dist<B>(rd_query, ptr_c);
        centroid_dist[i].second = i;
        ptr_c += B;
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + numC);

    // ===========================================================================================================
    // Scan the first nprobe clusters.
    Result *ptr_centroid_dist = (&centroid_dist[0]);
    uint8_t  PORTABLE_ALIGN64 byte_query[B];   
    
    for(int pb=0;pb<nprobe;pb++){ 
        uint32_t c = ptr_centroid_dist -> second;
        float sqr_y = ptr_centroid_dist -> first;
        ptr_centroid_dist ++;
        
        // =======================================================================================================
        // Preprocess the residual query and the quantized query
        float vl, vr;
        space.range(rd_query, centroid + c * B, vl, vr);
        float width = (vr - vl) / ((1 << B_QUERY) - 1);
        uint32_t sum_q = 0;
        space.quantize(byte_query, rd_query, centroid + c * B, u, vl, width, sum_q);
        
#if defined(SCAN)               // Binary String Representation 
        uint64_t PORTABLE_ALIGN32 quant_query[B_QUERY * B / 64];
        memset(quant_query, 0, sizeof(quant_query));
        space.transpose_bin(byte_query, quant_query);
#elif defined(FAST_SCAN)        // Look-Up-Table Representation 
        uint8_t PORTABLE_ALIGN32 LUT[B / 4 * 16];
        pack_LUT<B>(byte_query, LUT);
#endif
        
#if defined(SCAN)        
        scan(KNNs, distK, k,\
                quant_query, binary_code + start[c] * (B / 64), len[c], fac + start[c], \
                sqr_y, vl, width, sum_q,\
                query, data + start[c] * D, id + start[c]);
#elif defined(FAST_SCAN)
        fast_scan(KNNs, distK, k, \
                LUT, packed_code + packed_start[c], len[c], fac + start[c], \
                sqr_y, vl, width, sum_q,\
                query, data + start[c] * D, id + start[c]);
#endif
    }
    return KNNs;
}




// ==============================================================================================================================
// Save and Load Functions
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::save(char * filename){
    std::ofstream output(filename, std::ios::binary);

    uint32_t d = D;
    uint32_t b = B;
    output.write((char *) &N, sizeof(uint32_t));
    output.write((char *) &d, sizeof(uint32_t));
    output.write((char *) &C, sizeof(uint32_t));
    output.write((char *) &b, sizeof(uint32_t));

    output.write((char *) start     , C * sizeof(uint32_t));
    output.write((char *) len       , C * sizeof(uint32_t));
    output.write((char *) id        , N * sizeof(uint32_t));
    output.write((char *) dist_to_c , N * sizeof(float));
    output.write((char *) x0        , N * sizeof(float));

    output.write((char *) centroid, C * B * sizeof(float));
    output.write((char *) data, N * D * sizeof(float));
    output.write((char *) binary_code, N * B / 64 * sizeof(uint64_t));
    
    output.close();
    std::cerr << "Saved!" << std::endl;
}

// load impl
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::load(char * filename){
    std::ifstream input(filename, std::ios::binary);
    //std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &N, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t));
    input.read((char *) &C, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));

    std::cerr << d << std::endl;
    assert(d == D);
    assert(b == B);

    u = new float [B];
#if defined(RANDOM_QUERY_QUANTIZATION)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for(int i=0;i<B;i++)u[i] = uniform(gen);
#else 
    for(int i=0;i<B;i++)u[i] = 0.5;
#endif 

    centroid     = new float [C * B];
    data         = new float [N * D];

    binary_code  = static_cast<uint64_t*>(aligned_alloc(256, N * B / 64 * sizeof(uint64_t)));

    start        = new uint32_t [C];
    len          = new uint32_t [C];
    id           = new uint32_t [N];
    dist_to_c    = new float [N];
    x0           = new float [N];

    fac          = new Factor[N];

    input.read((char *) start      , C * sizeof(uint32_t));
    input.read((char *) len        , C * sizeof(uint32_t));
    input.read((char *) id         , N * sizeof(uint32_t));
    input.read((char *) dist_to_c  , N * sizeof(float));
    input.read((char *) x0         , N * sizeof(float));
    
    input.read((char *) centroid   , C * B * sizeof(float));
    input.read((char *) data, N * D * sizeof(float));
    input.read((char *) binary_code, N * B / 64 * sizeof(uint64_t));

#if defined(FAST_SCAN)
    packed_start = new uint32_t [C];
    int cur = 0;
    for(int i=0;i<C;i++){
        packed_start[i] = cur;
        cur += (len[i] + 31) / 32 * 32 * B / 8;
    }
    packed_code = static_cast<uint8_t*>(aligned_alloc(32, cur * sizeof(uint8_t)));
    for(int i=0;i<C;i++){
        pack_codes<B>(binary_code + start[i] * (B / 64), len[i], packed_code + packed_start[i]);
    }
#else 
    packed_start = NULL;
    packed_code  = NULL;
#endif
    for(int i=0;i<N;i++){
        long double x_x0 = (long double) dist_to_c[i] / x0[i];
        fac[i].sqr_x = dist_to_c[i] * dist_to_c[i];
        fac[i].error = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c[i] * dist_to_c[i]);
        fac[i].factor_ppc = -2 / fac_norm * x_x0  * ((float)space.popcount(binary_code + i * B / 64) * 2 - B);
        fac[i].factor_ip = -2 / fac_norm * x_x0;
    }
    input.close();
}


// ==============================================================================================================================
// Construction and Deconstruction Functions
template <uint32_t D, uint32_t B>
IVFRN<D, B>::IVFRN(){
    N = C = 0;
    start = len = id = NULL;
    x0 = dist_to_c = centroid = data = NULL;
    binary_code = NULL;
    fac = NULL;
    u = NULL;
}

template <uint32_t D, uint32_t B>
IVFRN<D, B>::IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid, 
            const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary){
    fac=NULL;
    u = NULL;

    N = X.n;
    C = _centroids.n;
    
    // check uint64_t
    assert(B % 64 == 0);
    assert(B >= D);

    start = new uint32_t [C];
    len   = new uint32_t [C];
    id    = new uint32_t [N];
    dist_to_c = new float [N];
    x0 = new float [N];

    memset(len, 0, C * sizeof(uint32_t));
    for(int i=0;i<N;i++)len[cluster_id.data[i]] ++;
    int sum = 0;
    for(int i=0;i<C;i++){
        start[i] = sum;
        sum += len[i];
    }
    for(int i=0;i<N;i++){
        id[start[cluster_id.data[i]]] = i;
        dist_to_c[start[cluster_id.data[i]]] = dist_to_centroid.data[i];
        x0[start[cluster_id.data[i]]] = _x0.data[i];
        start[cluster_id.data[i]]++;
    }
    for(int i=0;i<C;i++){
        start[i] -= len[i];
    }

    centroid        = new float [C * B];
    data            = new float [N * D];
    binary_code     = new uint64_t [N * B / 64];

    std::memcpy(centroid, _centroids.data, C * B * sizeof(float));
    float * data_ptr = data;
    uint64_t * binary_code_ptr = binary_code;

    for(int i=0;i<N;i++){
        int x = id[i];
        std::memcpy(data_ptr, X.data + x * D, D * sizeof(float));
        std::memcpy(binary_code_ptr, binary.data + x * (B / 64), (B / 64) * sizeof(uint64_t));
        data_ptr += D;
        binary_code_ptr += B / 64;
    }
}

template <uint32_t D, uint32_t B>
IVFRN<D, B>::~IVFRN(){
    if(id != NULL)          delete [] id;
    if(dist_to_c != NULL)   delete [] dist_to_c;
    if(len != NULL)         delete [] len;
    if(start != NULL)       delete [] start;
    if(x0 != NULL)          delete [] x0;
    if(data != NULL)        delete [] data;
    if(fac != NULL)         delete [] fac;
    if(u  != NULL)          delete [] u;
    if(binary_code != NULL) std::free(binary_code);
    // if(pack_codes  != NULL) std::free(pack_codes);
    if(centroid != NULL)    std::free(centroid);
}
