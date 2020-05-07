/* Multiplicação de matrizes utilizando Cuda 
* Código originalmente retirado de: http://www.muriloboratto.docentes.uneb.br/arquivos/cad/docs/codigos/handson/add-parallel-cudaMemcpy.cu
* Última modificação: 27/04/2019
* por gabri14el [gabri14el@gmail.com]
*/

#include <cuda.h>
#include <math.h>
#include "util.c"

#define BLOCKDIM 16

__global__ void saxpy(int n,  float *x, float *y){

 int i = threadIdx.x;

 if(i < n)
   y[i] = x[i] + y[i];

}

__global__ void multiply(int n, double *m, double *n, double *c){
  //pega os dados 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0;
  //verifica se valores e col
  if(row < n && col < n){
    sum = 0;
    //faz o dot product
    for (int i = 0; i < n; i++){
      sum += m[row * n + i] * n[i * n + col];
    }
    //salva em c
    c[row*n + col] = sum;
  }
}


void printVector(float *vector, int n){

for (int i=0; i < n ; ++i)
 printf("%1.0f\t", vector[i]);
 printf("\n\n");

}



void generateVector(float *vector, int n){

for (int i=0; i < n ; ++i)
 vector[i] = i + 1;

}


int main(int argc, char *argv[]){

  int n = atoi(argv[1]);




  //alocacao dos vetores no host
  double *h_m = generateMatrix(n , 100);
  double *h_n = generateMatrix(n , 100);
  double *h_c = (double *) malloc(n*n*sizeof(double));
  
  //alocacao dos vetores do device
  double *d_m, *d_n, *d_c;
  cudaMalloc((void**)&d_m, sizeof(double) * n *n );
  cudaMalloc((void**)&d_n, sizeof(double) * n * n );
  cudaMalloc((void**)&d_c, sizeof(double) * n * n );
  
  //copia dos vetores para o device
  cudaMemcpy(d_m, h_m, sizeof(double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_n, sizeof(double) * n, cudaMemcpyHostToDevice);

  //criacao das dimensoes
  //bloco
  dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);
  //grid
  int tam = (int) ceil(((float)n)/BLOCKDIM);
  dim3 gridDim(tam, tam, 1);

  //invacao da funcao
  multiply<<gridDim, blockDim>>(n, d_m, d_n, d_c);

  //copia do resultado
  cudaMemcpy(h_c, d_c, sizeof(double) * (n*n), cudaMemcpyDeviceToHost);

  //printVector(y, n);

  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_c);

  free(h_m);
  free(h_n);

  return 0;

}

