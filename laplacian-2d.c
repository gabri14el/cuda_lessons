/*%****************************************************************************80
!  Code: 
!   laplacian-2d.c
!
!  Purpose:
!   Implements sample 2D Laplacian Method in C code.
!
!  Modified:
!   Jan 20 2019 10:57 
!
!  Author:
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!   Murilo Boratto  <muriloboratto 'at' gmail.com>
!
!  How to Compile:
!   gcc laplacian-2d.c -o object
!
!  Execute: 
!   ./object
!
!  Comments: 
!   1) Simple Testbed with size problem n = m = 8. 
!   2) module load gcc/7.3.0
!   3) module load intel-xe-2018/2018
!   4) srun -p largemem gcc laplacian-2d.c -o object
!   5) srun -p largemem ./object                                                 
!     
!****************************************************************************80*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void kernel(double *a, double *c,  int m, int n, int jsta2, int jend2, int dx, int dz) {

  double sx, sz;
  int j, i;

  for(j = jsta2 - 1; j < jend2; j++){
     for(i = 1; i < (m - 1); i++){
      sx = a[(i-1) + j*n] + a[(i+1)+ j*n] + 2 * a[i + j*n];
      sz = a[ i + (j-1)*n] + a[i + (j+1)*n] + 2 * a[i + j*n];
      c[i + j * n] = (sx/(dx*dx)) + (sz/(dz*dz));
     }
  }

}/*kernel*/

void show_matrix(double *a, int n){

   int i, j;

   for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        printf("%1.2f\t", a[i + j*n]);
      }
    printf("\n");
   }

   printf("\n");

}/*show_matrix*/

void show_vector(int *a, int n){

   int i;

   for(i = 0; i < n; i++)
     printf("%d\t", a[i]);

   printf("\n\n");

}/*show_vector*/

void PARA_RANGE_1(int n1,int n2, int nprocs, int myid, int jsta, int jend, int *vector_return){

	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	jsta   = (myid * iwork1) + n1 + fmin((double)myid, (double)iwork2);
	jend   = jsta + iwork1 - 1;

	if (iwork2 > myid)
	 jend = jend + 1;

  vector_return[0] = jsta;
  vector_return[1] = jend;

} /*PARA_RANGE_1*/

int main(int argc, char *argv[])
{

        int n, m;
        int myid;
        int nprocs;
        double dx, dz;
        double *a, *c;
        int jsta = 1, jend = 1, jsta2, jend2;
        int i, j;
        int *vector_return = (int *) calloc (2, sizeof(int));
     
        n = m = 8; /*size problem*/

        a  =  (double*) calloc (m * n, sizeof(double));
        c  =  (double*) calloc (m * n, sizeof(double));

        dx = 1; dz = 1;

	      myid = 0;
        nprocs = 1;

	      PARA_RANGE_1(1, n, nprocs, myid, jsta, jend, vector_return);

	      jsta = vector_return[0];
	      jend = vector_return[1];

        jsta2 = jsta;
	      jend2 = jend;

        jsta2 = 2;         
        jend2 = n - 1;
	
        /*Population of the matrix*/
	      for (i = 0; i < m; i++)
	        for(j = jsta-1; j < jend; j++)
	           a[i + j* n] = (i + j + 2) * 1.;  /*stored dates in column (major column)*/

        show_matrix(a, n);
       
               kernel(a, c, m, n, jsta2, jend2, dx, dz);
   
        show_matrix(c, n); 
       
        /*Free memories*/
        free(a);
        free(c);
     
        return 0;

}/*main*/
