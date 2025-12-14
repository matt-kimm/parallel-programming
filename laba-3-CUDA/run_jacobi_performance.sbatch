#!/bin/bash
#SBATCH --job-name=cuda_jacobi_perf
#SBATCH --output=cuda_perf.%j.out
#SBATCH --error=cuda_perf.%j.err
#SBATCH --partition=compclass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32G

echo "=== CUDA Jacobi Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo ""

# Настройка CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "=== Compiling ==="
nvcc -o cuda_jacobi_performance cuda_jacobi_performance.cu -O2

if [ $? -eq 0 ]; then
    echo "? Compilation successful!"
    echo ""
    
    echo "=== Generating performance table ==="
    echo "This will test:"
    echo "  Matrix sizes: 100, 500, 1000, 2000"
    echo "  Threads per block: 128, 256, 512"
    echo ""
    
    # Запускаем генерацию таблицы
    ./cuda_jacobi_performance
    
    echo ""
    echo "=== Quick verification tests ==="
    
    # Быстрые проверочные тесты
    echo "Test 1: 1000x1000 matrix, 256 threads"
    ./cuda_jacobi_performance 1000 256
    
    echo ""
    echo "Test 2: 500x500 matrix, 128 threads"
    ./cuda_jacobi_performance 500 128
    
    echo ""
    echo "=== Generated files ==="
    ls -la performance_table.txt performance_results.csv 2>/dev/null || echo "No output files yet"
    
else
    echo "? Compilation failed!"
    echo "Trying with architecture flag..."
    nvcc -o cuda_jacobi_performance cuda_jacobi_performance.cu -O2 -arch=sm_75
    if [ $? -eq 0 ]; then
        echo "? Compilation successful with arch flag!"
        ./cuda_jacobi_performance
    fi
fi

echo ""
echo "=== Job completed ==="
echo "End: $(date)"