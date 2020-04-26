#include <stdio.h>
#include <omp.h>
#include <stdlib.h>


void CreateMyMatrices();
int ** allocateContig(int m, int n);

int **MatrixA,**MatrixB, **ResultMat, rows,cols,i,j,k;

int main() {
    int id=0, np=0;
    CreateMyMatrices();

#pragma omp parallel shared(MatrixA,MatrixB,ResultMat, rows, cols) private(i)
    {

        #pragma omp for schedule(static)
        for (i=0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                ResultMat[i][j] = MatrixA[i][j] + MatrixB[i][j];
                printf("in thread %d result[%d][%d] = %d \n", omp_get_thread_num(), i, j, ResultMat[i][j]);
            }
        }
    }   /* end of parallel region */

    for (k=0; k<rows; k++)
    {
        for(i=0; i<cols; i++)
        {
            printf("%d ", ResultMat[k][i]);
        }
        printf("\n");
    }
}


void CreateMyMatrices(){
    printf("Enter matrices rows and cols numbers : ");
    scanf("%d",&rows);
    scanf("%d",&cols);


    MatrixA = allocateContig(rows,cols);
    MatrixB = allocateContig(rows,cols);
    ResultMat = allocateContig(rows,cols);

    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            MatrixA[i][j]=0.0;
            MatrixB[i][j]=0.0;
            ResultMat[i][j]=0.0;
        }
    }

    printf("Enter First Matrix Elements =\n");
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            scanf("%d",&MatrixA[i][j]);
        }
    }

    printf("Enter Second Matrix Elements =\n");
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            scanf("%d",&MatrixB[i][j]);
        }
    }
}

int ** allocateContig(int m, int n){
    int *linear,**mat,i;
    linear = malloc(sizeof(int)*m *n);
    mat = malloc(sizeof(int*)*m);
    for(i = 0; i<m; i++) {
        mat[i] = &linear[i * n];
    }
    return  mat;
}