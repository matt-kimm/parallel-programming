#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <map>
#include <cuda_runtime.h>

using namespace std;

// Ядро для вычисления новой итерации (метод Якоби)
__global__ void jacobi_kernel(
    const double* A,
    const double* b,
    const double* x_old,
    double* x_new,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                sum += A[i * N + j] * x_old[j];
            }
        }
        x_new[i] = (b[i] - sum) / A[i * N + i];
    }
}

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << \
            " - " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Функция для запуска одного эксперимента
double run_experiment(int N, int threads, bool verbose = false) {
    double tol = 1e-6;
    int max_iter = 1000;
    
    // Инициализация данных
    vector<double> A(N * N);
    vector<double> b(N);
    vector<double> x(N, 0.0);
    
    // Заполнение матрицы (диагонально доминирующая)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? (2.0 * N + 1.0) : 1.0;
        }
        b[i] = (2.0 * N + 1.0) + (N - 1) * 1.0;
    }
    
    // Выделение памяти на GPU
    double *d_A, *d_b, *d_x_old, *d_x_new;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x_old, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x_new, N * sizeof(double)));
    
    // Копирование данных на GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x_old, x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    int blocks = (N + threads - 1) / threads;
    
    // Замер времени
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    
    // Итерационный процесс
    vector<double> x_new_host(N);
    vector<double> x_old_host(N);
    
    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x_old, d_x_new, N);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Host reduction для вычисления нормы
        CHECK_CUDA_ERROR(cudaMemcpy(x_new_host.data(), d_x_new, N * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(x_old_host.data(), d_x_old, N * sizeof(double), cudaMemcpyDeviceToHost));
        
        double residual = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = x_new_host[i] - x_old_host[i];
            residual += diff * diff;
        }
        residual = sqrt(residual);
        
        if (residual < tol) {
            if (verbose) {
                cout << "  Converged after " << iter << " iterations" << endl;
            }
            break;
        }
        
        // Обмен указателей
        swap(d_x_old, d_x_new);
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    double execution_time = gpu_time_ms / 1000.0;
    
    // Освобождение памяти
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_x_old));
    CHECK_CUDA_ERROR(cudaFree(d_x_new));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return execution_time;
}

// Функция для вывода таблицы Time(s) | Speedup | Efficiency(%)
void print_performance_table() {
    vector<int> matrix_sizes = {100, 500, 1000, 2000};
    vector<int> thread_configs = {128, 256, 512};
    
    cout << "\n" << string(80, '=') << endl;
    cout << "PERFORMANCE TABLE: Time(s) | Speedup | Efficiency(%)" << endl;
    cout << string(80, '=') << endl;
    
    // Для каждого размера матрицы выводим отдельную таблицу
    for (int N : matrix_sizes) {
        if (N > 2000) continue; // Пропускаем большие матрицы если нет памяти
        
        cout << "\nMatrix Size: " << N << "x" << N << endl;
        cout << string(60, '-') << endl;
        cout << left << setw(10) << "Threads" 
             << setw(15) << "Time(s)" 
             << setw(12) << "Speedup" 
             << setw(15) << "Efficiency(%)" << endl;
        cout << string(60, '-') << endl;
        
        // Запускаем эксперименты для всех конфигураций потоков для ЭТОЙ матрицы
        vector<double> times;
        for (int threads : thread_configs) {
            double time = run_experiment(N, threads, false);
            times.push_back(time);
        }
        
        // Находим лучшее время для ЭТОЙ матрицы
        double best_time = *min_element(times.begin(), times.end());
        
        // Выводим результаты для ЭТОЙ матрицы
        for (size_t i = 0; i < thread_configs.size(); i++) {
            int threads = thread_configs[i];
            double time = times[i];
            double speedup = best_time / time;  // Ускорение относительно лучшего времени для ЭТОЙ матрицы
            double efficiency = (speedup / threads) * 100.0 * 128;  // Нормализуем к 128 потокам
            
            cout << left << setw(10) << threads
                 << fixed << setprecision(6) << setw(15) << time
                 << setprecision(3) << setw(12) << speedup
                 << setprecision(2) << setw(15) << efficiency << endl;
        }
        cout << string(60, '-') << endl;
    }
    
    // Сохраняем таблицу в файл
    ofstream table_file("performance_table.txt");
    if (table_file.is_open()) {
        table_file << "CUDA Jacobi Solver Performance Table\n";
        table_file << "Time(s) | Speedup | Efficiency(%)\n";
        table_file << string(60, '-') << "\n";
        
        for (int N : matrix_sizes) {
            if (N > 2000) continue;
            
            table_file << "\nMatrix Size: " << N << "x" << N << "\n";
            table_file << string(60, '-') << "\n";
            table_file << left << setw(10) << "Threads" 
                      << setw(15) << "Time(s)" 
                      << setw(12) << "Speedup" 
                      << setw(15) << "Efficiency(%)" << "\n";
            table_file << string(60, '-') << "\n";
            
            // Запускаем эксперименты для файла
            vector<double> times;
            for (int threads : thread_configs) {
                double time = run_experiment(N, threads, false);
                times.push_back(time);
            }
            
            double best_time = *min_element(times.begin(), times.end());
            
            for (size_t i = 0; i < thread_configs.size(); i++) {
                int threads = thread_configs[i];
                double time = times[i];
                double speedup = best_time / time;
                double efficiency = (speedup / threads) * 100.0 * 128;
                
                table_file << left << setw(10) << threads
                          << fixed << setprecision(6) << setw(15) << time
                          << setprecision(3) << setw(12) << speedup
                          << setprecision(2) << setw(15) << efficiency << "\n";
            }
            table_file << string(60, '-') << "\n";
        }
        table_file.close();
        cout << "\nTable saved to: performance_table.txt" << endl;
    }
    
    // Сохраняем в CSV
    ofstream csv_file("performance_results.csv");
    if (csv_file.is_open()) {
        csv_file << "MatrixSize,Threads,Time(s),Speedup,Efficiency(%)\n";
        
        for (int N : matrix_sizes) {
            if (N > 2000) continue;
            
            vector<double> times;
            for (int threads : thread_configs) {
                double time = run_experiment(N, threads, false);
                times.push_back(time);
            }
            
            double best_time = *min_element(times.begin(), times.end());
            
            for (size_t i = 0; i < thread_configs.size(); i++) {
                int threads = thread_configs[i];
                double time = times[i];
                double speedup = best_time / time;
                double efficiency = (speedup / threads) * 100.0 * 128;
                
                csv_file << N << "," << threads << "," 
                        << fixed << setprecision(6) << time << ","
                        << setprecision(3) << speedup << ","
                        << setprecision(2) << efficiency << "\n";
            }
        }
        csv_file.close();
        cout << "CSV data saved to: performance_results.csv" << endl;
    }
}

int main(int argc, char* argv[]) {
    cout << "=== CUDA Jacobi Solver Performance Analysis ===" << endl;
    
    // Получение информации о GPU
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    cout << "\nGPU Information:" << endl;
    cout << "  Name: " << prop.name << endl;
    cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    
    if (argc >= 3) {
        // Режим запуска с параметрами
        int N = atoi(argv[1]);
        int threads = atoi(argv[2]);
        
        cout << "\nSingle experiment:" << endl;
        cout << "  Matrix: " << N << "x" << N << endl;
        cout << "  Threads: " << threads << endl;
        
        double time = run_experiment(N, threads, true);
        
        cout << "\nResults:" << endl;
        cout << "  Time(s): " << fixed << setprecision(6) << time << endl;
        
        // Вычисляем blocks
        int blocks = (N + threads - 1) / threads;
        cout << "  Blocks: " << blocks << endl;
        cout << "  Total threads: " << blocks * threads << endl;
        
    } else {
        // Режим генерации таблицы
        cout << "\nGenerating performance table..." << endl;
        cout << "Matrix sizes: 100, 500, 1000, 2000" << endl;
        cout << "Thread configurations: 128, 256, 512" << endl;
        
        print_performance_table();
    }
    
    cout << "\n=== Analysis completed ===" << endl;
    return 0;
}
