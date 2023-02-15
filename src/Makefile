#Makefile 
#define variables
objects= main.o timer.o 
NVCC= /usr/local/cuda-11/bin/nvcc       #cuda c compiler
opt= -use_fast_math -O2 -g -G           #optimization flag
ARCH=-arch=sm_50 \ 			#cuda compute capability
-gencode=arch=compute_50,code=sm_50 \ 
-gencode=arch=compute_52,code=sm_52 \ 
-gencode=arch=compute_60,code=sm_60 \ 
-gencode=arch=compute_61,code=sm_61 \ 
-gencode=arch=compute_70,code=sm_70 \ 
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_75,code=compute_75 
execname = main
L1CACHE_DISABLE = -Xptxas -dlcm=cg
SEGM_LEN = $(segm_len)

#compile
$(execname): $(objects)
	#$(NVCC) $(opt) $(L1CACHE_DISABLE) -o $(execname) $(objects) $(LIBS)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS)

timer.o: timer.cu
	$(NVCC) $(opt) $(ARCH) -c timer.cu
main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cu -D SEGMENT_LEN=$(SEGM_LEN)


#clean Makefile
clean:
	rm $(objects)

#end of Makefile

