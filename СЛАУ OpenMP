#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

int main() {
    int N = 100;          // размер системы
    double tol = 1e-6;    // точность
    int max_iter = 10000; // максимум итераций

    // Инициализация матрицы A
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    for (int i = 0; i < N; i++)
        A[i][i] = 2 * N + 1;

    // Выбираем некоторый вектор x для построения b = A*x (исходный вектор)
    vector<double> x(N, 1.0);
    vector<double> b(N, 0.0);

    // Вычисляем b = A*x параллельно
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++)
            s += A[i][j] * x[j];
        b[i] = s;
    }

    // "Забываем" x и начинаем итерационный процесс с нулевого вектора
    fill(x.begin(), x.end(), 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;

#pragma omp parallel for reduction(max:max_diff)
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i)
                    sum += A[i][j] * x[j];
            }
            double x_new = (b[i] - sum) / A[i][i];
            double diff = fabs(x_new - x[i]);
            if (diff > max_diff) max_diff = diff;
            x[i] = x_new;  // обновление "на месте"
        }

        if (max_diff < tol) {
            cout << "Converged after " << iter << " iterations with error " << max_diff << endl;
            break;
        }
    }

    //cout << "Результат (первые 10 элементов): ";
    //for (int i = 0; i < min(10, N); i++)
    //    cout << x[i] << " ";
    //cout << endl;

    return 0;
}
