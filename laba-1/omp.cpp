#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <map>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 10000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    double tol = 1e-6;
    int max_iter = 10000;
    
    // Замер времени выполнения
    double start_time, end_time, local_time, max_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Распределение строк между процессами
    int rows_per_proc = N / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_proc;
    int local_rows = end_row - start_row;
    
    // Инициализация матрицы A-строк локально и векторов
    vector<vector<double>> A_local(local_rows, vector<double>(N, 1.0));
    for (int i = 0; i < local_rows; i++) {
        A_local[i][start_row + i] = 2 * N + 1;
    }
    
    // Выбираем изначальный вектор x (на всех процессах полный)
    vector<double> x(N, 1.0);
    
    // Вычисляем вектор b = A*x (на всех процессах считаем локальную часть)
    vector<double> b_local(local_rows, 0.0);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            b_local[i] += A_local[i][j] * x[j];
        }
    }
    
    // Собираем b у процесса 0, затем рассылаем всем
    vector<double> b(N, 0.0);
    
    // Подготовка данных для MPI_Allgatherv
    vector<int> recvcounts(size);
    vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        int proc_start = i * rows_per_proc;
        int proc_end = (i == size - 1) ? N : proc_start + rows_per_proc;
        recvcounts[i] = proc_end - proc_start;
        displs[i] = proc_start;
    }
    
    MPI_Allgatherv(b_local.data(), local_rows, MPI_DOUBLE,
                   b.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    // "Забываем" x и начинаем итеративный метод для решения Ax = b
    fill(x.begin(), x.end(), 0.0);
    vector<double> x_new(local_rows, 0.0);
    vector<double> x_global(N, 0.0);  // Глобальный вектор для всех процессов
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Собираем полное решение со всех процессов
        MPI_Allgatherv(x_new.data(), local_rows, MPI_DOUBLE,
                       x_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Обновляем локальную часть x_new по формуле простой итерации
        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            int global_i = start_row + i;
            for (int j = 0; j < N; j++) {
                if (j != global_i) {
                    sum += A_local[i][j] * x_global[j];
                }
            }
            x_new[i] = (b[global_i] - sum) / A_local[i][global_i];
        }
        
        // Проверка сходимости
        double local_diff = 0.0;
        for (int i = 0; i < local_rows; i++) {
            int global_i = start_row + i;
            double diff = x_new[i] - x_global[global_i];
            local_diff += diff * diff;
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
    
    // Синхронизируем и замеряем окончательное время
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    local_time = end_time - start_time;
    
    // Находим максимальное время среди всех процессов
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Сохраняем результат текущего запуска
        ofstream outfile("performance_results.txt", ios::app);
        if (outfile.is_open()) {
            outfile << size << " " << N << " " << max_time << endl;
            outfile.close();
            cout << "\nSaved result: " << size << " processes, N=" << N 
                 << ", time=" << max_time << "s" << endl;
        }
        
        // Читаем ВСЕ сохраненные результаты
        ifstream infile("performance_results.txt");
        map<int, double> results;  // key: количество процессов, value: время
        
        int proc_count;
        int matrix_size;
        double time_val;
        
        while (infile >> proc_count >> matrix_size >> time_val) {
            if (matrix_size == N) {  // Только для текущего размера матрицы
                // Если уже есть запись для этого количества процессов,
                // берем минимальное время (лучший результат)
                if (results.find(proc_count) == results.end() || 
                    time_val < results[proc_count]) {
                    results[proc_count] = time_val;
                }
            }
        }
        infile.close();
        
        // Добавляем текущий результат (если еще нет)
        if (results.find(size) == results.end() || 
            max_time < results[size]) {
            results[size] = max_time;
        }
        
        // Выводим таблицу
        cout << "\n\nPerformance Table for N = " << N << ":\n";
        cout << "Threads              N     Time, s      Speedup S     Efficiency, %\n";
        cout << "--------------------------------------------------------------------\n";
        
        // Находим время для 1 процесса
        double time_1proc = -1.0;
        if (results.find(1) != results.end()) {
            time_1proc = results[1];
        }
        
        // Сортируем результаты по количеству процессов
        vector<pair<int, double>> sorted_results;
        for (const auto& res : results) {
            sorted_results.push_back(res);
        }
        sort(sorted_results.begin(), sorted_results.end());
        
        // Выводим таблицу
        cout << fixed << setprecision(5);
        for (const auto& res : sorted_results) {
            int procs = res.first;
            double time = res.second;
            
            cout << setw(7) << procs 
                 << setw(15) << N 
                 << setw(12) << time;
            
            if (time_1proc > 0) {
                double speedup = (procs == 1) ? 1.0 : time_1proc / time;
                double efficiency = (speedup / procs) * 100.0;
                
                cout << setw(12) << setprecision(2) << speedup
                     << setw(17) << setprecision(1) << efficiency;
            } else {
                cout << setw(12) << "N/A" << setw(17) << "N/A";
            }
            cout << endl;
        }
        
        // Также сохраняем в CSV
        ofstream csvfile("performance_results.csv");
        if (csvfile.is_open()) {
            csvfile << "Threads,N,Time(s),Speedup,Efficiency(%)\n";
            for (const auto& res : sorted_results) {
                int procs = res.first;
                double time = res.second;
                
                if (time_1proc > 0) {
                    double speedup = (procs == 1) ? 1.0 : time_1proc / time;
                    double efficiency = (speedup / procs) * 100.0;
                    csvfile << procs << "," << N << "," << time << "," 
                            << speedup << "," << efficiency << "\n";
                } else {
                    csvfile << procs << "," << N << "," << time << ",N/A,N/A\n";
                }
            }
            csvfile.close();
            cout << "\nResults also saved to performance_results.csv" << endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
