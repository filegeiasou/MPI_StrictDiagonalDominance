#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
#define N 4

struct MyStruct {
    int value;
    int i;
}in, out;

int main(int argc , char **argv)
{
    int rank,size;
    int min_val,min_i,min_j;
    int local_min;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N][N];
    int i,j;

    int *local_A;

    local_A = (int *)malloc(N * sizeof(int));

    if(rank == 0)
    {
        printf("Enter the elements of the matrix A:\n");
        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                scanf("%d", &A[i][j]);
            }
        }
    }

    MPI_Scatter(A, N, MPI_INT, local_A, N, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received row: ", rank);
    for (i = 0; i < N; i++) 
    {
        printf("%d ", local_A[i]);
    }
    printf("\n");

    int local_rows = N / size;
    int start_row = rank * local_rows;
    int end_row = start_row + local_rows;

    int is_strictly_diagonally_dominant = 1;
    for (int i = start_row; i < end_row; i++) 
    {
        int sum = 0;
        for (int j = 0; j < N; j++) 
        {
            if (i != j) 
            {
                sum += abs(local_A[j]);
            }
        }
        if (abs(local_A[i]) <= sum) 
        {
            is_strictly_diagonally_dominant = 0;
            break;
        }   
    }
    

    int global_result;
    MPI_Reduce(&is_strictly_diagonally_dominant, &global_result, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        if (global_result == 1) 
        {
            printf("yes\n");
            printf("The matrix is strictly diagonally dominant\n");
        } else {
            printf("no\n");
            printf("The matrix is not strictly diagonally dominant\n");
        }
    }
    
    if(global_result==1)
    {
        int local_max = 0;
        for (int i = start_row; i < end_row; i++) 
        {
            if (local_A[i] > local_max) 
            {
                local_max = local_A[i];
            }
        }

        int global_max;
        MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) 
        {
            printf("The maximum absolute value of the diagonal elements is %d\n", global_max);
        }
        int B[N][N];

        int* gather_buffer = NULL;
        if (rank == 0) 
        {
            gather_buffer = malloc(N * N * sizeof(int));
        }

        MPI_Gather(local_A, N, MPI_INT, gather_buffer, N, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) 
        {
            for (int i = 0; i < N; i++) 
            {
                for (int j = 0; j < N; j++) 
                {
                    if (i == j) 
                    {
                        B[i][j] = global_max;
                    } 
                    else 
                    {
                        B[i][j] = global_max - abs(gather_buffer[i * N + j]);
                    }
                }
            }

            printf("Matrix B:\n");
            for (int i = 0; i < N; i++) 
            {
                for (int j = 0; j < N; j++) 
                {
                    printf("%d ", B[i][j]);
                }
                printf("\n");
            }
        }
        int *local_B = (int *)malloc(N * sizeof(int));

        MPI_Scatter(B, N, MPI_INT, local_B, N, MPI_INT, 0, MPI_COMM_WORLD);
        local_min=INT_MAX;
        in.value = local_B[0];
        in.i = 0;
        for(int i=0; i<N ; i++)
        {
            if(local_B[i]<in.value)
            {
                in.value=local_B[i];
                in.i=i;
            }
        }    
        in.i=in.i+rank*N;
        MPI_Reduce(&in, &out, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
        if(rank==0)
        {
            min_val=out.value;
            min_i=out.i/N;
            min_j=out.i%N;
            printf("The minimum value of B is %d and it is located at (%d,%d)\n",min_val,min_i,min_j);
        }
    }
    MPI_Finalize();
    return 0;
}
