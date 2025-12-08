#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>  // для setw и setprecision

using namespace std;

int main() {
    int N = 500;          // размер системы
    double tol = 1e-6;    // точность
    int max_iter = 10000; // максимум итераций

    //vector<double> A(N*N);
    //A[i*N + j]
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    for (int i = 0; i < N; i++)
        A[i][i] = 2 * N + 1;

    vector<double> x(N, 1.0);
    vector<double> b(N, 0.0);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++)
            s += A[i][j] * x[j];
        b[i] = s;
    }

    int threads_list[] = { 1, 2, 4, 8, 16 };
    double times[5] = { 0 };
    int iterations_list[5] = { 0 }; 

    for (int t = 0; t < 5; t++) {
        int num_threads = threads_list[t];
        omp_set_num_threads(num_threads);
        std::cerr << "numtr = " << omp_get_num_threads() << "\n";
        fill(x.begin(), x.end(), 0.0);

        double start_time = omp_get_wtime();

        int iter = 0;
        double final_max_diff = 0.0;

        for (iter = 0; iter < max_iter; iter++) {
            double max_diff = 0.0;

#pragma omp parallel for reduction(max:max_diff) //schedule(dynamic:5)
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++) {
                    if (j != i)
                        sum += A[i][j] * x[j];
                }
                double x_new = (b[i] - sum) / A[i][i];
                double diff = fabs(x_new - x[i]);
                if (diff > max_diff) max_diff = diff;
                x[i] = x_new;
            }

            if (max_diff < tol) {
                final_max_diff = max_diff;
                iterations_list[t] = iter + 1; 
                break;
            }

            // Если не сошлось за max_iter итераций
            if (iter == max_iter - 1) {
                iterations_list[t] = max_iter;
            }
        }

        double end_time = omp_get_wtime();
        times[t] = end_time - start_time;

        // Если не было достигнуто условие сходимости
        if (iterations_list[t] == 0) {
            iterations_list[t] = iter;
        }
    }

    // Вывод таблицы
    cout << setw(8) << "Threads"
        << setw(12) << "N"
        << setw(12) << "Time, s"
        << setw(15) << "Speedup S"
        << setw(18) << "Efficiency, %"
        << setw(15) << "Iterations" << endl;

    cout << string(75, '-') << endl;

    for (int t = 0; t < 5; t++) {
        double speedup = times[0] / times[t];
        double efficiency = speedup / threads_list[t] * 100.0;

        cout << setw(8) << threads_list[t]
            << setw(12) << N
            << setw(12) << fixed << setprecision(5) << times[t]
            << setw(15) << fixed << setprecision(2) << speedup
            << setw(17) << fixed << setprecision(1) << efficiency << "%"
            << setw(15) << iterations_list[t]
            << endl;
    }

    cout << "\n N = " << N
        << ", tol = " << tol
        << ", max_iter = " << max_iter << endl;

    return 0;
}
