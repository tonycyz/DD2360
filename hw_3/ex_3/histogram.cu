#include <stdio.h>
#include <sys/time.h>
#include <random>

//#define NUM_BINS 4096


#define NUM_BINS 1024

#define TPB 32
#define TPB2 32


__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  int idx= blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>num_elements) return;

  __shared__ unsigned int shared_bins[NUM_BINS];//shared memory



  if(threadIdx.x == 0) {
      for (int i = 0; i < NUM_BINS; i++)
          shared_bins[i] = 0;//initialize,in every block, set shared_bins =0
    }

  __syncthreads();

    atomicAdd(&shared_bins[input[idx]], 1);

  __syncthreads();

  if(threadIdx.x == 0) {
  for (int i =0;i<NUM_BINS;i++){
    atomicAdd(&bins[i],shared_bins[i]);//get back results in each block
  }
  }
}



__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int bin_id= blockIdx.x * blockDim.x + threadIdx.x;
  if (bin_id>num_bins) return;
  if (bins[bin_id]>127) bins[bin_id]=127;

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*)malloc(sizeof(unsigned int) * inputLength);
  hostBins = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BINS);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)

  for (int i=0;i<inputLength;i++){
    hostInput[i] = rand()%NUM_BINS;
  }


  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int*)malloc(NUM_BINS * sizeof(int));
  for (int i=0;i<inputLength;i++){
    resultRef[i] = 0;
  }

  for (int i=0;i<inputLength;i++){
    resultRef[hostInput[i]]=resultRef[hostInput[i]]+1;
  }

  for (int i=0;i<NUM_BINS;i++){
    if(resultRef[hostInput[i]]>127)
      resultRef[hostInput[i]]=127;
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc((void**)&deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc((void**)&deviceBins, sizeof(unsigned int) * NUM_BINS);


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  //@@ Initialize the grid and block dimensions here



  dim3 dimBlock(TPB,1,1);
  dim3 dimGrid((inputLength+TPB-1)/TPB,1,1);
  //@@ Launch the GPU Kernel here
  histogram_kernel<<<dimGrid,dimBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here

  dim3 dimBlock2(TPB2,1,1);
  dim3 dimGrid2((inputLength+TPB2-1)/TPB2,1,1);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<dimGrid2,dimBlock2>>>(deviceBins,NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  for (int i=0;i<NUM_BINS;i++){
    printf("%d, ",hostBins[i]);

  }
  printf("printend\n");
  for (int i=0;i<NUM_BINS;i++){

    printf("%d, ",resultRef[i]);
  }
  printf("printend\n");

  int notequal = 0;
  for(int i=0;i<NUM_BINS;i++){
      if( hostBins[i] != resultRef[i] ) {
          notequal = 1;
          break;
      }
  }

  if (notequal == 1){
      printf("The result is different from refernce!");
  }
  else if (notequal == 0){
      printf("The result aligns with the refernce!");
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}