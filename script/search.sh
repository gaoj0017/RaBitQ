


source='./data'
data='sift'
C=4096
B=128
D=128
k=100

g++ -march=core-avx2 -Ofast -o ./bin/search_${data} ./src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

result_path=./results
mkdir ${result_path}

res="${result_path}/${data}/"

mkdir "$result_path/${data}/"

./bin/search_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"
