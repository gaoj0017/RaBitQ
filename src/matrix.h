
#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>
#include <Eigen/Dense>

template <typename T>
class Matrix
{
private:

public:
    T* data;
    size_t n;
    size_t d;

    // Construction
    Matrix(); // Default
    Matrix(size_t n, size_t d); // Fixed size
    Matrix(const Matrix<T> &X); // Deep Copy
    Matrix(size_t n, size_t d, const Matrix<T> &X); // Fixed size with a filling matrix.
    Matrix(size_t n, size_t d, const Matrix<T> &X, size_t *id); // Submatrix with given row numbers.
    Matrix(const Matrix<T> &X, const Matrix<size_t> &ID); // Submatrix
    Matrix(const Matrix<T> &X, const size_t id); // row
    Matrix(char * data_file_path); // IO
    Matrix(size_t n); // ID

    // Deconstruction
    ~Matrix(){
        delete [] data;
    }

    // Serialization
    void serialize(FILE * fp);
    void deserialize(FILE * fp);

    Matrix & operator = (const Matrix &X){
        delete [] data;
        n = X.n;
        d = X.d;
        data = new T [n*d];
        memcpy(data, X.data, sizeof(T) * n * d);
        return *this;
    }

    // Linear Algebra
    void add(size_t a, const Matrix<T> &B, size_t b);
    void div(size_t a, T c);
    void mul(const Matrix<T> &A, Matrix<T> &result) const;
    void copy(size_t r, const Matrix<T> &A, size_t ra); // copy the ra th row of A to r th row
    float dist(const Matrix<T> &A, size_t a, const Matrix<T> &B, size_t b) const;
    float dist(size_t a, const Matrix<T> &B, size_t b) const;

    size_t scalar(){
        return data[0];
    }

    bool empty(){
        if(n == 0)return 1;
        return 0;
    }

    // Experiment and Debug
    void print();
    void reset();
    
};

template <typename T>
Matrix<T>::Matrix(){
    n = 0;
    d = 0;
    data = NULL;
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& X){
    n = X.n;
    d = X.d;
    data = new T[n * d];
    memcpy(data, X.data, sizeof(T) * n * d);
}

template <typename T>
Matrix<T>::Matrix(size_t _n,size_t _d){
    n = _n;
    d = _d;
    data = new T [n * d + 10];
    memset(data, 0, (n * d + 10) * sizeof(T));
}

template <typename T>
Matrix<T>::Matrix(size_t _n,size_t _d, const Matrix<T> &X){
    n = _n;
    d = _d;
    data = new T [n * d + 10];
    memset(data, 0, (n * d + 10) * sizeof(T));
    for(int i=0;i<n;i++){
        memcpy(data + i * d, X.data + i * X.d, sizeof(T) * X.d);
    }
}

template <typename T>
Matrix<T>::Matrix(size_t _n,size_t _d, const Matrix<T> &X, size_t *id){
    n = _n;
    d = _d;
    data = new T [n * d];
    for(size_t i=0;i<n;i++){
        memcpy(data + i * d, X.data + id[i] * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &X, const Matrix<size_t> &id){
    n = id.n;
    d = X.d;
    data = new T [n * d];
    for(size_t i=0;i<n;i++){
        memcpy(data + i * d, X.data + id.data[i] * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &X, const size_t id){
    n = 1;
    d = X.d;
    data = new T [n * d];
    for(size_t i=0;i<n;i++){
        memcpy(data + i * d, X.data + id * d, sizeof(T) * d);
    }
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path){
    n = 0;
    d = 0;
    data = NULL;
    printf("%s\n",data_file_path);
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&d, 4);
    
    std::cerr << "Dimensionality - " <<  d <<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (sizeof(T) * d + 4));
    // n = (size_t)(fsize / (d + 1) / 4);
    data = new T [(size_t)n * (size_t)d + 10];
    std::cerr << "Cardinality - " << n << std::endl;
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * d), d * sizeof(T));
    }
    in.close();
}

template <typename T>
Matrix<T>::Matrix(size_t _n){
    n = _n;
    d = 1;
    data = new T [n];
    for(size_t i=0;i<n;i++) data[i] = i;
}

template <typename T>
void Matrix<T>::print(){
    for(size_t i=0;i<2;i++){
        std::cout << "(";
        for(size_t j=0;j<d;j++){
            std::cout << data[i * d + j] << (j == d-1 ? ")":", ");
        }
        std:: cout << std::endl;
    }
}

template <typename T>
void Matrix<T>::reset(){
    memset(data, 0, sizeof(T) * n * d);
}

template <typename T>
float Matrix<T>::dist(const Matrix<T> &A, size_t a, const Matrix<T> &B, size_t b)const{
    float dist = 0;
    float *ptra = A.data + a * d;
    float *ptrb = B.data + b * d;

    for(int i=0;i<d;i++){
        float t = *ptra - *ptrb;
        dist += t * t;
        ptra ++;
        ptrb ++;
    }
    
    return dist;
}

template <typename T>
float Matrix<T>::dist(size_t a, const Matrix<T> &B, size_t b)const{
    float dist = 0;
    for(size_t i=0;i<d;i++){
        dist += (data[a * d + i] - B.data[b * d + i]) * (data[a * d + i] - B.data[b * d + i]);
    }   
    return dist;
}

template <typename T>
void Matrix<T>::add(size_t a, const Matrix<T> &B, size_t b){
    for(size_t i=0;i<d;i++)
        data[a * d + i] += B.data[b * d + i];
}


template <typename T>
void Matrix<T>::div(size_t a, T c){
    for(size_t i=0;i<d;i++)
        data[a * d + i] /= c;
}


template <typename T>
void Matrix<T>::mul(const Matrix<T> &A, Matrix<T> &result)const{
    //result.reset();
    result.n = n;
    result.d = A.d;
    for(size_t i=0;i<n;i++){
        for(size_t k=0;k<A.d;k++){
            T p=0;
            for(size_t j=0;j<d;j++){
                p += data[i * d + j] * A.data[j * A.d + k];
            }
            result.data[i * A.d + k] = p;
        }
    }
}

template<typename T>
Matrix<T> mul(const Matrix<T> &A, const Matrix<T> &B){

    std::cerr << "Matrix Multiplication - " << A.n << " " << A.d << " " << B.d << std::endl;
    Eigen::MatrixXd _A(A.n, A.d);
    Eigen::MatrixXd _B(B.n, B.d);
    Eigen::MatrixXd _C(A.n, B.d);
    
    for(int i=0;i<A.n;i++)
        for(int j=0;j<A.d;j++)
            _A(i,j)=A.data[i*A.d+j];
    
    for(int i=0;i<B.n;i++)
        for(int j=0;j<B.d;j++)
            _B(i,j)=B.data[i*B.d+j];

    _C = _A * _B;

    Matrix<T> result(A.n, B.d);

    for(int i=0;i<A.n;i++)
        for(int j=0;j<B.d;j++)
            result.data[i*B.d+j] = _C(i,j);
    
    return result;
}

template <typename T>
void Matrix<T>::serialize(FILE * fp){
    fwrite(&n, sizeof(size_t), 1, fp);
    fwrite(&d, sizeof(size_t), 1, fp);
    size_t size = sizeof(T);
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(data, size, n * d, fp);
}

template <typename T>
void Matrix<T>::deserialize(FILE * fp){
    fread(&n, sizeof(size_t), 1, fp);
    fread(&d, sizeof(size_t), 1, fp);
    //std::cerr << n << " " << d << std::endl;
    assert(n <= 1000000000);
    assert(d <= 2000);

    size_t size = sizeof(T);
    fread(&size, sizeof(size_t), 1, fp);
    data = new T [n * d];
    fread(data, size,  n * d, fp);
}

double normalize(float *x, unsigned D){
    Eigen::VectorXd v(D);
    for(int i=0;i<D;i++)v(i)=x[i];
    
    double norm = v.norm();
    v.normalize();
    for(int i=0;i<D;i++)x[i]=v(i);
    return norm;
}


#endif
