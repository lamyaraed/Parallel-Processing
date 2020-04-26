#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

void CreateMyMatrices();
int ** allocateContig(int m, int n);

int **MatrixA,**MatrixB, **ResultMat, Arows,Acols,Brows,Bcols,i,j,k;
int main(int argc , char * argv[]) {
    //int **MatrixA,**MatrixB,Arows,Acols,Brows,Bcols,i,j;

    int my_rank;        /* rank of process	*/
    int p;            /* number of process	*/
    int source;        /* rank of sender	*/
    int dest;        /* rank of reciever	*/
    int tag = 0;        /* tag for messages	*/
    char message[100];    /* storage for message	*/
    int index;
    MPI_Status status;    /* return status for 	recieve		*/
    MPI_Request request; /*Capture Request for Send*/

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of process */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank != 0) { //Slaves cores
        dest = 0; //master core
        int RowsToSlaves;
        int MatrixSize;
        int RemainderRows;

        source = 0;
        MPI_Recv(&Acols, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&RowsToSlaves, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&MatrixSize, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&Bcols, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&Brows, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

        if(my_rank==(p-1)){
            MPI_Recv(&RemainderRows, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

            MatrixA = allocateContig(RowsToSlaves+RemainderRows,Acols);
            MatrixB = allocateContig(Brows,Bcols);

            MPI_Recv(&(MatrixA[0][0]), Acols*RowsToSlaves, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&(MatrixB[0][0]), Brows*Bcols , MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&(MatrixA[RowsToSlaves][0]), RemainderRows*Acols , MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            RowsToSlaves +=RemainderRows;
            MatrixSize = (RowsToSlaves*Bcols); //matrix size changes here as last core has more size than others
        }
        else
        {
            MatrixA = allocateContig(RowsToSlaves,Acols);
            MatrixB = allocateContig(Brows,Bcols);

            MPI_Recv(&(MatrixA[0][0]), Acols*RowsToSlaves, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&(MatrixB[0][0]), Brows*Bcols , MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        }
        ResultMat = allocateContig(RowsToSlaves,Bcols);

        for (i = 0; i< RowsToSlaves; i++) //initializes result matrix to zero
        {
            for (j=0; j< Bcols; j++){
                ResultMat[i][j] = 0;
            }
        }


        for (i = 0; i <RowsToSlaves; i++) {//iterate through a given set of rows of [A]
            for (j = 0; j <Bcols; j++) {//iterate through columns of [B]
                for (k = 0; k <Brows; k++) {//iterate through rows of [B]
                    ResultMat[i][j] += (MatrixA[i][k] * MatrixB[k][j]);
                }
            }
        }

        for (i = 0; i < RowsToSlaves; i++){
            for (j = 0; j < Bcols; j++)
                printf("%d ", ResultMat[i][j]);
            printf("\n");
        }

        MPI_Send(&RowsToSlaves, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        MPI_Send(&(ResultMat[0][0]), MatrixSize, MPI_INT, dest, tag, MPI_COMM_WORLD);

    }
    else{ // Master core (my_rank ==0)

        int dest;
        int rNum = 0;

        printf("Welcome To Matrix Multiplication Program.\n");
        printf("I am the Master core!\n");
        CreateMyMatrices();

        int RowsToSlaves = Arows/(p-1);
        int RemainderRows = Arows%(p-1);
        int MatrixSize= RowsToSlaves*Bcols; //change here
        int startrow=0;

        for(dest=1; dest<p; dest++){

            MPI_Send(&Acols, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            MPI_Send(&RowsToSlaves, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            MPI_Send(&MatrixSize, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            MPI_Send(&Bcols, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            MPI_Send(&Brows, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);


            if(dest==(p-1)) {
                MPI_Send(&RemainderRows, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&(MatrixA[startrow][0]), RowsToSlaves*Acols , MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&(MatrixB[0][0]), Brows*Bcols , MPI_INT, dest, tag, MPI_COMM_WORLD);
                startrow +=RowsToSlaves;
                MPI_Send(&(MatrixA[startrow][0]), RemainderRows * Acols, MPI_INT, dest, tag, MPI_COMM_WORLD);
            }
            else{
                MPI_Send(&(MatrixA[startrow][0]), RowsToSlaves*Acols , MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&(MatrixB[0][0]), Brows*Bcols , MPI_INT, dest, tag, MPI_COMM_WORLD);

                startrow +=RowsToSlaves;
            }
        }

        printf("Your Resulting Matrix is : \n");
        ResultMat = allocateContig(Arows,Bcols); //changed to A rows and B columns
        startrow =0;

        MatrixSize = RowsToSlaves * Bcols;
        for(dest=1; dest<p; dest++)
        {
            MPI_Recv(&RowsToSlaves, 1, MPI_INT, dest, tag, MPI_COMM_WORLD,&status);
            MatrixSize = RowsToSlaves*Bcols;
            MPI_Recv(&(ResultMat[startrow][0]), MatrixSize, MPI_INT, dest, tag, MPI_COMM_WORLD,&status);
            startrow +=RowsToSlaves;
        }

        int row,columns;
        for (row=0; row<Arows; row++) //Arows instead of Brows
        {
            for(columns=0; columns<Bcols; columns++)
            {
                printf("%d ", ResultMat[row][columns]);
            }
            printf("\n");
        }
    }

    MPI_Finalize(); //added


    return 0;
}


void CreateMyMatrices(){
    printf("Enter 1st matrix rows and cols numbers : ");
    scanf("%d",&Arows);
    scanf("%d",&Acols);

    printf("Enter 2nd matrix rows and cols numbers : ");
    scanf("%d %d",&Brows, &Bcols);

    while(Brows!=Acols){
        printf("The two matrices can not be multiplied!" "\n Enter another dimensions  : ");

        printf("Enter 1st matrix rows and cols numbers : ");
        scanf("%d %d",&Arows,&Acols);

        printf("Enter 2nd matrix rows and cols numbers : ");
        scanf("%d %d",&Brows, &Bcols);
    }

    /*MatrixA =(int **)malloc(Arows*sizeof(int *));
    for(i=0; i<Arows; i++)
        MatrixA[i]=(int *)malloc(Acols*sizeof(int ));

    MatrixB =(int **)malloc(Brows*sizeof(int *));
    for(i=0; i<Brows; i++)
        MatrixB[i]=(int *)malloc(Bcols*sizeof(int ));
*/

    MatrixA = allocateContig(Arows,Acols);
    MatrixB = allocateContig(Brows,Bcols);

    printf("Enter First Matrix Elements =\n");
    for(i=0;i<Arows;i++){
        for(j=0;j<Acols;j++){
            scanf("%d",&MatrixA[i][j]);
        }
    }

    printf("Enter Second Matrix Elements =\n");
    for(i=0;i<Brows;i++){
        for(j=0;j<Bcols;j++){
            scanf("%d",&MatrixB[i][j]);
        }
    }
}

int ** allocateContig(int m, int n){
    int *linear,**mat,i;
    linear = malloc(sizeof(int)*m*n);
    mat = malloc(sizeof(int*)*m);
    for(i = 0; i<m; i++) {
        mat[i] = &linear[i * n];
    }
    return  mat;
}

