
#include <iostream>
#include <cstdlib>
#include <mpi.h>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10000;
    int count = 0, total = 0;

    srand(3655942 + rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    for (int i = rank; i < n; i += size) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (x*x + y*y <= 1.0)
            count++;
    }

    MPI_Reduce(&count, &total, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    t = MPI_Wtime() - t;

    if (rank == 0) {
        double pi = (double)total / n * 4;

        cout << "Time: " << t << endl;
        cout << "no. of processes: " << size << endl;
        cout << "no. of trials: " << n << endl;
        cout << "est. of pi: " << pi << endl;
    }

    MPI_Finalize();
    return 0;
}

