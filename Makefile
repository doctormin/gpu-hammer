CUDA_HOME	?= /usr/local/cuda

NVCC		:= $(CUDA_HOME)/bin/nvcc
NVFLAGS		:= -std=c++14 -g -O3 -Xcompiler -Wall

.PHONY: all clean

all: bin/gpu_hammer

SRC := $(wildcard src/*.cu)
OBJ := $(patsubst src/%.cu,build/%.o,$(SRC))

bin/gpu_hammer: $(OBJ)
	@mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) -o $@ $+

build/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) -o $@ -c $<

clean:
	rm -rf build bin
