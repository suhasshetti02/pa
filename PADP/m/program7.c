
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 400

int main() {
    int n = SIZE * SIZE;

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    double *d = malloc(n * sizeof(double));

    if (!a || !b || !c || !d) {
        printf("Memory allocation failed\n");
        free(a); free(b); free(c); free(d);
        return 1;
    }

    
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            a[i*SIZE+j] = i+j;
            b[i*SIZE+j] = i-j;
            c[i*SIZE+j] = d[i*SIZE+j] = 0;
        }

    struct timeval tim;
    gettimeofday(&tim, NULL);
    double t1 = tim.tv_sec + tim.tv_usec / 1000000.0;

    
    #pragma acc parallel loop collapse(2) copyin(a[0:n], b[0:n]) copyout(c[0:n])
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            double sum = 0;

            #pragma acc loop seq
            for (int k = 0; k < SIZE; k++)
                sum += a[i*SIZE+k] * b[k*SIZE+j];

            c[i*SIZE+j] = sum;
        }

    gettimeofday(&tim, NULL);
    double t2 = tim.tv_sec + tim.tv_usec / 1000000.0;

    printf("%.6lf seconds with OpenACC\n", t2-t1);

    
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            double sum = 0;

            for (int k = 0; k < SIZE; k++)
                sum += a[i*SIZE+k] * b[k*SIZE+j];

            d[i*SIZE+j] = sum;
        }

    
    for (int i = 0; i < n; i++)
        if (c[i] != d[i]) {
            printf("Error at index %d\n", i);
            free(a); free(b); free(c); free(d);
            return 1;
        }

    printf("OpenACC matrix multiplication test was successful!\n");

    free(a); free(b); free(c); free(d);
    return 0;
}

