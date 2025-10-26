#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100; // Размер матрицы
    double tol = 1e-6; // Порог сходимости
    int max_iter = 10000;

    // Распределение строк между процессами
    int rows_per_proc = N / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_proc;

    // Инициализация матрицы A-строк локально и векторов
    vector<vector<double>> A_local(end_row - start_row, vector<double>(N, 1.0));
    for (int i = 0; i < end_row - start_row; i++) {
        A_local[i][start_row + i] = 2 * N + 1;
    }

    // Выбираем изначальный вектор x (на всех процессах полный)
    vector<double> x(N, 1.0);

    // Вычисляем вектор b = A*x (на всех процессах считаем локальную часть)
    vector<double> b_local(end_row - start_row, 0.0);
    for (int i = 0; i < end_row - start_row; i++) {
        for (int j = 0; j < N; j++) {
            b_local[i] += A_local[i][j] * x[j];
        }
    }

    // Собираем b у процесса 0, затем рассылаем всем (для простоты)
    vector<double> b(N, 0.0);
    MPI_Allgather(b_local.data(), end_row - start_row, MPI_DOUBLE,
        b.data(), end_row - start_row, MPI_DOUBLE, MPI_COMM_WORLD);

    // "Забываем" x и начинаем итеративный метод для решения Ax = b
    fill(x.begin(), x.end(), 0.0);

    vector<double> x_new(end_row - start_row, 0.0);
    vector<double> x_old(N, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
        // Сохраняем предыдущее глобальное решение
        MPI_Allgather(x_new.data(), end_row - start_row, MPI_DOUBLE,
            x_old.data(), end_row - start_row, MPI_DOUBLE, MPI_COMM_WORLD);

        // Обновляем локальную часть x_new по формуле простой итерации
        for (int i = 0; i < end_row - start_row; i++) {
            double sum = 0.0;
            int global_i = start_row + i;
            for (int j = 0; j < N; j++) {
                if (j != global_i) {
                    sum += A_local[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[global_i] - sum) / A_local[i][global_i];
        }

        // Проверка сходимости на процессе 0
        double local_diff = 0.0;
        for (int i = 0; i < end_row - start_row; i++) {
            local_diff += (x_new[i] - x_old[start_row + i]) * (x_new[i] - x_old[start_row + i]);
        }
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_diff = sqrt(global_diff);

        if (rank == 0 && iter % 100 == 0) {
            cout << "Iteration " << iter << ", error = " << global_diff << endl;
        }

        if (global_diff < tol) {
            if (rank == 0) {
                cout << "Converged after " << iter << " iterations" << endl;
            }
            break;
        }
    }
    
    //if (rank == 0) {
    //    cout << "Solution x (partial view):" << endl;
    //    for (int i = 0; i < min(N, 10); i++) {
    //        cout << x_new[i] << " ";
    //    }
    //    cout << endl;
    //}
    
    MPI_Finalize();
    return 0;
}
