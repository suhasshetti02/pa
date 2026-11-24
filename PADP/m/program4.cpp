
#include <iostream>
#include <vector>
#include <cstdio>
#include <gd.h>
#include <omp.h>
using namespace std;

int main() {
    FILE *fp = fopen("input.png", "rb");
    if (!fp) {
        cout << "Image not found\n";
        return 1;
    }

    gdImagePtr img = gdImageCreateFromPng(fp);
    fclose(fp);

    if (!img) return 1;

    int w = gdImageSX(img);
    int h = gdImageSY(img);

    vector<int> input(w*h), gray(w*h), owner(w*h);

    // Read image pixels
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int c = gdImageGetPixel(img, x, y);

            int r = gdImageRed(img, c);
            int g = gdImageGreen(img, c);
            int b = gdImageBlue(img, c);

            input[y*w+x] = (r+g+b)/3;
        }

    omp_set_num_threads(4);
    omp_set_dynamic(0);

    int chunks[] = {1, 10, 50};
    const char *names[] = {"Static", "Dynamic", "Guided"};

    // Part A: Scheduling comparison
    for (int s = 0; s < 3; s++) {
        for (int chunk : chunks) {

            if (s == 0)
                omp_set_schedule(omp_sched_static, chunk);
            else if (s == 1)
                omp_set_schedule(omp_sched_dynamic, chunk);
            else
                omp_set_schedule(omp_sched_guided, chunk);

            double start = omp_get_wtime();

            #pragma omp parallel for schedule(runtime)
            for (int x = 0; x < w; x++) {
                int tid = omp_get_thread_num();

                for (int y = 0; y < h; y++) {
                    int i = y*w+x;
                    gray[i] = input[i];
                    owner[i] = tid;
                }
            }

            double time = omp_get_wtime() - start;

            cout << names[s] << " Chunk: " << chunk
                 << " Time: " << time << " s\n";
        }
    }

    // Part B: Show thread colors
    omp_set_schedule(omp_sched_static, 10);

    #pragma omp parallel for schedule(runtime)
    for (int x = 0; x < w; x++) {
        int tid = omp_get_thread_num();

        for (int y = 0; y < h; y++)
            owner[y*w+x] = tid;
    }

    gdImagePtr bw = gdImageCreateTrueColor(w, h);
    gdImagePtr threads = gdImageCreateTrueColor(w, h);

    int colors[] = {
        gdTrueColor(255, 0, 0),
        gdTrueColor(0, 255, 0),
        gdTrueColor(0, 0, 255),
        gdTrueColor(0, 0, 0)
    };

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int i = y*w+x;
            int g = gray[i];

            gdImageSetPixel(bw, x, y,
                            gdTrueColor(g, g, g));

            gdImageSetPixel(threads, x, y,
                            colors[owner[i]]);
        }

    fp = fopen("gray.png", "wb");
    if (!fp) return 1;
    gdImagePng(bw, fp);
    fclose(fp);

    fp = fopen("threads.png", "wb");
    if (!fp) return 1;
    gdImagePng(threads, fp);
    fclose(fp);

    gdImageDestroy(img);
    gdImageDestroy(bw);
    gdImageDestroy(threads);

    cout << "Images saved successfully\n";
    return 0;
}

