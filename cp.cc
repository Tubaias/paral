#include <math.h>
#include <iostream>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double avgI = 0.0;
            double avgJ = 0.0;

            for (int x = 0; x < nx; x++) {
                avgI += data[x + i*nx];
                avgJ += data[x + j*nx];
            }

            avgI /= nx;
            avgJ /= nx;

            double top = 0.0;
            double bot1 = 0.0;
            double bot2 = 0.0;

            for (int x = 0; x < nx; x++) {
                double elemI = data[x + i*nx] - avgI;
                double elemJ = data[x + j*nx] - avgJ;

                top += elemI * elemJ;
                bot1 += elemI * elemI;
                bot2 += elemJ * elemJ;
            }

            double bot = sqrt(bot1 * bot2);
            result[i + j*ny] = float(top / bot);
        }
    }
}

/*
void correlate(int ny, int nx, const float *data, float *result) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sumI, sumJ, sqsumI, sqsumJ, sumIJ = 0;

            for (int k = 0; k < nx; k++) {
                double elemI = data[k + i*nx];
                double elemJ = data[k + j*nx];

                sumI += elemI;
                sumJ += elemJ;
                sqsumI += elemI * elemI;
                sqsumJ += elemJ * elemJ;
                sumIJ += elemI * elemJ;
            }

            std::cout << "sumI: " << sumI << std::endl;
            std::cout << "sumJ: " << sumJ << std::endl;
            std::cout << "sumIJ: " << sumIJ << std::endl;

            float busd = (float)(nx * sumIJ - sumI * sumJ) / sqrt((nx * sqsumI - sumI * sumI) * (nx * sqsumJ - sumJ * sumJ));
            std::cout << "busd: " << busd << std::endl;
            result[i + j*ny] = busd;
        }
    }
}
*/

/*
void correlate(int ny, int nx, const float *data, float *result) {
    // normalization
    float norData[nx * ny];
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            norData[x + y*nx] = data[x + y*nx];
        }
    }

    for (int row = 0; row < ny; row++) {
        double sum = 0.0;
        double sqsum = 0.0;

        for (int col = 0; col < nx; col++) {
            double elem = double(norData[col + row*nx]);
            sum += elem;
            sqsum += elem * elem;
        }

        for (int col = 0; col < nx; col++) {
            norData[col + row*nx] -= sum / nx;
            norData[col + row*nx] /= sqsum;
        }
    }

    result = norData;
}
*/