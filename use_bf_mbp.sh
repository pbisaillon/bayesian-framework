#GTEST
export LD_LIBRARY_PATH=$PWD/unit-tests/gtest-1.7.0/build:$LD_LIBRARY_PATH

#ARMADILLO
export LD_LIBRARY_PATH=$PWD/armadillo/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/armadillo/usr/local/lib64:$LD_LIBRARY_PATH
#LIBCONFIG
export LD_LIBRARY_PATH=$PWD/libconfig/lib:$LD_LIBRARY_PATH

export BFPATH=$(pwd)
