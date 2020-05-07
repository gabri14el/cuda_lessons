/* Multiplicação de matrizes utilizando Cuda 
* Código originalmente retirado de: http://www.muriloboratto.docentes.uneb.br/arquivos/cad/docs/codigos/handson/add-parallel-cudaMemcpy.cu
* Última modificação: 27/04/2019
* por gabri14el [gabri14el@gmail.com]
*/

#include <cuda.h>
#include <math.h>
#include "util.c"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define BLOCKDIM 16
#define TILE_WIDTH 16

__global__ void multiply(int tam, double *m, double *n, double *c){
  //pega os dados 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0;
  int i;
  //verifica se valores e col
  if(row < tam && col < tam){
    sum = 0;
    //faz o dot product
    for (i = 0; i < tam; i++){
      sum += m[row*tam + i] * n[i*tam + col];
    }
    //salva em c
    c[row*tam + col] = sum;
  }
}

__global__ void tiledMultiply(int tam, double *m, double *n, double *c){
  
  //cria vetores na shared memory
  __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

  //captura as coordenadas para selecionar os dados 
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //ainda preciso explicar isso aqui
  //coluna e linha do elemento que estamos calculando no momento
  int row = by* TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  double c_value = 0; 

  //variavel ph corresponde a fase que esta sendo executado no momento
  for(int ph = 0; ph < ceil(tam/(float)TILE_WIDTH); ph++){
    if((row < tam) && (ph*TILE_WIDTH+tx)<tam)
      Mds[ty][tx] = m[row*tam + ph*TILE_WIDTH + tx];
    if(col < tam && (ph*TILE_WIDTH+ty) < tam)
      Nds[ty][tx] = n[(ph*TILE_WIDTH + ty)*tam + col];
    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; k++){
      c_value += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if(row < tam && col < tam)
    c[row*tam + col] = c_value;

}




int main(int argc, char *argv[]){

  int n = atoi(argv[1]);
  int nn = n*n;
  //alocacao dos vetores no host
  printf("alocando e gerando matrizes no host..\n");
  
  double *h_m = generateMatrix(n , 100);
  double *h_n = generateMatrix(n , 100);
  double *h_c = (double *) malloc(nn*sizeof(double));
  
  //alocacao dos vetores do device
  double *d_m, *d_n, *d_c;
  printf("alocando memória no device..\n");
  cudaMalloc((void**)&d_m, sizeof(double) * nn);
  cudaMalloc((void**)&d_n, sizeof(double) * nn);
  cudaMalloc((void**)&d_c, sizeof(double) * nn);
  cudaCheckErrors("cudamalloc fail");
  
  printf("copiando dados para o device..\n");
  //copia dos vetores para o device
  cudaMemcpy(d_m, h_m, sizeof(double) * nn, cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, h_n, sizeof(double) * nn, cudaMemcpyHostToDevice);
  //inicializa 'c' com zeros
  cudaMemset(d_c, 0, nn * sizeof(double));
  cudaCheckErrors("cuda memcpy fail");

  //criacao das dimensoes
  //bloco
  /*dim3 blockDim(BLOCKDIM, BLOCKDIM);
  //grid
  int tam = (int) ceil(((float)n)/BLOCKDIM);
  dim3 gridDim(tam, tam);*/

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  int tam = (n - 1)/TILE_WIDTH + 1;
  dim3 gridDim(tam, tam);
  

  //invacao da funcao
  printf("executando kernel..\n");
  //multiply<<<gridDim, blockDim>>>(n, d_m, d_n, d_c);
  tiledMultiply<<<gridDim, blockDim>>>(n, d_m, d_n, d_c);
  cudaDeviceSynchronize();


  //copia do resultado
  printf("kernel finalizado, copiando resultado para o host..\n");
  cudaMemcpy(h_c, d_c, sizeof(double) * nn, cudaMemcpyDeviceToHost);
  //cudaThreadSynchronize();
  cudaCheckErrors("cudamemcpy or cuda kernel fail");


  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_c);

    /*
  printf("Matriz M: \n");
  printMatrix(h_m, n, n);
  
  printf("\n\nMatriz N: \n");
  printMatrix(h_n, n, n);
  
  printf("\n\nMatriz MxN: \n");
  printMatrix(h_c, n, n);	
  */
	
  free(h_m);
  free(h_n);
  free(h_c);

  return 0;
}

