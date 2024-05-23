
C=4096
data='sift'
D=128
B=128
source='./data'

g++ -o ./bin/index_${data} ./src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN

./bin/index_${data} -d $data -s "$source/$data/"    