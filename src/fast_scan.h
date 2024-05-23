
// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)
#pragma once
#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <assert.h>

#define lowbit(x) (x&(-x))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

using namespace std;

// ==============================================================
// look up the tables for a packed batch of 32 quantization codes
// ==============================================================
template <uint32_t B>
inline void accumulate_one_block(uint8_t* codes, uint8_t* LUT, uint16_t* result){
    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu[4];
    for(int i=0;i<4;i++){
        accu[i] = _mm256_setzero_si256();   
    }

    constexpr uint32_t M = B / 4;

    for(int m=0;m<M;m+=2){
        __m256i c   = _mm256_load_si256((__m256i const*)codes);
        __m256i lo  = _mm256_and_si256(c, low_mask);
        __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        __m256i lut = _mm256_load_si256((__m256i const*)LUT);

        __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

        accu[0] = _mm256_add_epi16(accu[0], res_lo);
        accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res_lo, 8));

        accu[2] = _mm256_add_epi16(accu[2], res_hi);
        accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res_hi, 8));
        
        codes += 32;
        LUT   += 32;
    }

    accu[0] = _mm256_sub_epi16(accu[0], _mm256_slli_epi16(accu[1], 8));
    __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[0], accu[1], 0x21),_mm256_blend_epi32(accu[0], accu[1], 0xF0));
    _mm256_store_si256((__m256i*)(result + 0 ), dis0);
    
    accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));
    __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[2], accu[3], 0x21),_mm256_blend_epi32(accu[2], accu[3], 0xF0));
    _mm256_store_si256((__m256i*)(result + 16), dis1);    
}

// ==============================================================
// look up the tables for all the packed batches
// ==============================================================
template <uint32_t B>
inline void accumulate(uint32_t nblk, uint8_t* codes, uint8_t* LUT, uint16_t* result){
    for(int i=0;i<nblk;i++){
        accumulate_one_block<B>(codes, LUT, result);
        codes  += 32 * B / 8;
        result += 32;
    }
}


// ==============================================================
// prepare the look-up-table from the quantized query vector
// ==============================================================
template <uint32_t B>
inline void pack_LUT(uint8_t* byte_query, uint8_t* LUT){
    constexpr uint32_t M = B / 4;
    constexpr uint32_t pos[16]={
        3 /*0000*/, 3/*0001*/, 2/*0010*/, 3/*0011*/,
        1 /*0100*/, 3/*0101*/, 2/*0110*/, 3/*0111*/,
        0 /*1000*/, 3/*1001*/, 2/*1010*/, 3/*1011*/,
        1 /*1100*/, 3/*1101*/, 2/*1110*/, 3/*1111*/,
    };
    for(int i=0;i<M;i++){
        LUT[0] = 0;
        for(int j=1;j<16;j++){
            LUT[j] = LUT[j - lowbit(j)] + byte_query[pos[j]];
        }
        LUT        += 16;
        byte_query += 4;
    }
}


template <typename T, class TA>
inline void get_matrix_column(T* src, size_t m, size_t n, int64_t i, int64_t j, TA& dest) {
    for (int64_t k = 0; k < dest.size(); k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } 
        else {
            dest[k] = 0;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization 
// codes represented by a sequence of uint8_t variables
// ==============================================================
template <uint32_t B>
void pack_codes(const uint8_t* codes, uint32_t ncode, uint8_t* blocks){
    
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    constexpr uint32_t M = B / 4;
    const uint8_t bbs = 32;    
    memset(blocks, 0, ncode_pad * M / 2);

    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    uint8_t* codes2 = blocks;
    for(int blk=0;blk<ncode_pad;blk+=bbs){
        // enumerate m
        for(int m=0;m<M;m+=2){
            std::array<uint8_t, 32> c, c0, c1;
            get_matrix_column(codes, ncode, M / 2, blk, m / 2, c);
            for (int j = 0; j < 32; j++) {
                c0[j] = c[j] & 15;
                c1[j] = c[j] >> 4;
            }
            for (int j = 0; j < 16; j++) {
                uint8_t d0, d1;
                d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                codes2[j] = d0;
                codes2[j + 16] = d1;
            }
            codes2 += 32;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization 
// codes represented by a sequence of uint64_t variables
// ==============================================================
template<uint32_t B>
void pack_codes(const uint64_t* binary_code, uint32_t ncode, uint8_t* blocks){
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    memset(blocks, 0, ncode_pad * sizeof(uint8_t));

    uint8_t * binary_code_8bit = new uint8_t [ncode_pad * B / 8];
    memcpy(binary_code_8bit, binary_code, ncode * B / 64 * sizeof(uint64_t));

    for(int i=0;i<ncode;i++)
        for(int j=0;j<B/64;j++)
            for(int k=0;k<4;k++)
                swap(binary_code_8bit[i * B / 8 + 8 * j + k], binary_code_8bit[i * B / 8 + 8 * j + 8 - k - 1]);

    for(int i=0;i<ncode*B/8;i++){
        uint8_t v = binary_code_8bit[i];
        uint8_t x = (v >> 4);
        uint8_t y = (v & 15);
        binary_code_8bit[i] = (y << 4 | x);
    }
    pack_codes<B>(binary_code_8bit, ncode, blocks);
    delete [] binary_code_8bit;
}
