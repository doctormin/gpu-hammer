CUDA_HOME	?= /usr/local/cuda

NVCC		:= $(CUDA_HOME)/bin/nvcc
NVFLAGS		:= -std=c++14 -g -O3 -Xcompiler -Wall
NVARCH		:= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70

.PHONY: all clean

all: bin/gpu_hammer

SRC := $(wildcard src/*.cu)
OBJ := $(patsubst src/%.cu,build/%.o,$(SRC))

l1:
	python makeRecipe.py l1

l2:
	python makeRecipe.py l2

gmem:
	python makeRecipe.py gmem

gmem_fp:
	python makeRecipe.py gmem_fp

fp32:
	python makeRecipe.py fp32

fp64:
	python makeRecipe.py fp64

fp:
	python makeRecipe.py fp
	
bin/gpu_hammer: $(OBJ)
	@mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) $(NVARCH) -o $@ $+

build/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) $(NVARCH) -o $@ -c $<

clean:
	rm -rf build bin
