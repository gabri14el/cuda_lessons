#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef unsigned long int ulint;


double randomDouble(double min, double max)
{
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

double *generateMatrix(ulint n, int max)
{
	double *M = malloc(n * n * sizeof(double));

	double v;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (max > 0)
				M[i * n + j] = rand() % max;
			else
			{
				v = randomDouble(-99999.0, 99999.0);
				M[i * n + j] = v;
				// Se v é zero, um, ou um inteiro, volta o j para gerá-lo novamente
				if (((int)v) == 0 || ((int)v) == 1 || ((int)v) == v)
					j--;
			}
		}
	}

	return M;
}