
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
using namespace std;

// 1. Cache-unfriendly sieve
int unfriendly(int n) {
    vector<char> mark(n + 1, 0);

    for (int i = 2; 1LL*i*i <= n; i++)
        if (!mark[i])
            for (long long j = 1LL*i*i; j <= n; j += i)
                mark[j] = 1;

    int count = 0;
    for (int i = 2; i <= n; i++)
        if (!mark[i]) count++;

    return count;
}

// 2. Cache-friendly and parallel sieve
int friendly(int n, bool parallel) {
    int root = sqrt(n);
    vector<char> base(root + 1, 0);

    // Find prime factors up to sqrt(n)
    for (int i = 2; i*i <= root; i++)
        if (!base[i])
            for (int j = i*i; j <= root; j += i)
                base[j] = 1;

    const int BLOCK = 10000;
    int blocks = (n - 2) / BLOCK + 1;
    int count = 0;

    #pragma omp parallel for if(parallel) reduction(+:count) schedule(dynamic)
    for (int b = 0; b < blocks; b++) {
        int low = 2 + b * BLOCK;
        int high = min(low + BLOCK - 1, n);

        vector<char> mark(high - low + 1, 0);

        for (int p = 2; p <= root; p++) {
            if (!base[p]) {
                long long start = max(1LL*p*p,
                    ((long long)low + p - 1) / p * p);

                for (long long j = start; j <= high; j += p)
                    mark[j - low] = 1;
            }
        }

        for (int i = low; i <= high; i++)
            if (!mark[i - low]) count++;
    }

    return count;
}

int main() {
    int n = 1000000;
    omp_set_num_threads(4);

    double t = omp_get_wtime();
    int a = unfriendly(n);
    double t1 = omp_get_wtime() - t;

    t = omp_get_wtime();
    int b = friendly(n, false);
    double t2 = omp_get_wtime() - t;

    t = omp_get_wtime();
    int c = friendly(n, true);
    double t3 = omp_get_wtime() - t;

    cout << "Cache-unfriendly primes: " << a << endl;
    cout << "Cache-friendly primes: " << b << endl;
    cout << "Parallel primes: " << c << endl;

    cout << "Cache-unfriendly time: " << t1 << " s\n";
    cout << "Cache-friendly time: " << t2 << " s\n";
    cout << "Parallel time: " << t3 << " s\n";

    cout << "Correctness: "
         << ((a == b && b == c) ? "PASS" : "FAIL")
         << endl;

    return 0;
}

