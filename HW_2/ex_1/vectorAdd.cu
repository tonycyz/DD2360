#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define DataType double
#define TPB 256
#define MAXTHREAD 1024

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here

  int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    if (id < len)
        out[id] = in1[id] + in2[id];

}


//@@ Insert code to implement timer start
double timestart(){
  struct timeval t1;
  gettimeofday(&t1, NULL);
  return (double) (1000000.0*(t1.tv_sec) + t1.tv_usec)/1000000.0;

}

//@@ Insert code to implement timer stop
double timestop(double t1){
  struct timeval t2;
  gettimeofday(&t2, NULL);
  return (double) (1000000.0*(t2.tv_sec) + t2.tv_usec)/1000000.0 - t1;
}


int main(int argc, char **argv) {
  
  int inputLength;
  double timecost;
  double start;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args

  printf("testing");
  
  inputLength = atoi (argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output

  hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));
  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

  for (int i=0;i<inputLength;i++){
      hostInput1[i] = rand()/(DataType)RAND_MAX;
      hostInput2[i] = rand()/(DataType)RAND_MAX;
      resultRef[i] = hostInput1[i] + hostInput2[i]; 
  }


  //@@ Insert code below to allocate GPU memory here



  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
  




  //@@ Insert code to below to Copy memory to the GPU here

  start = timestart();

  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  timecost = timestop(start);
  printf("Copying memory host to device cost %f seconds\n", timecost);



  //@@ Initialize the 1D grid and block dimensions here

  int gridnum = (inputLength+TPB-1)/TPB;
  int blocksize = TPB;


  //@@ Launch the GPU Kernel here

  double start1 = timestart();
  vecAdd<<<gridnum, blocksize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double timecost1 = timestop(start1);
  printf("launch the kernel cost %f seconds\n", timecost1);


  //@@ Copy the GPU memory back to the CPU here

  double start2 = timestart();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  double timecost2 = timestop(start2);
  printf("copy GPU memory to CPU cost %f seconds\n", timecost2);

  //@@ Insert code below to compare the output with the reference

  int notequal = 0;
  DataType diff = 1e-8;
  for(int i=0;i<inputLength;i++){
      if (abs(hostOutput[i] - resultRef[i]) > diff) {
          notequal = 1;
      }
  }

  if (notequal == 1){
      printf("The result is different from refernce!");
  }
  else if (notequal == 0){
      printf("The result aligns with the refernce!");
  }
    


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}