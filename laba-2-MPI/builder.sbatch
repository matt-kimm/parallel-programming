#!/bin/bash
#SBATCH -J sle_mpi
#SBATCH -p compclass
#SBATCH -o out/sle_mpi.%j.out
#SBATCH -e err/sle_mpi.%j.err
#SBATCH -N 4                # Number of nodes
#SBATCH -n 16                # Total number of MPI tasks
#SBATCH -t 12:00:00
#SBATCH --mem=4G

module load compilers/gcc-15.2.0
module load mpi/openmpi

# Compile
echo "Compiling MPI code..."
mpic++ -std=c++17 -O3 laba-MPI.cpp -o runner

# SLURM information
echo "--------------------------------------"
echo "SLURM allocated nodes : $SLURM_JOB_NUM_NODES"
echo "SLURM allocated tasks : $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "--------------------------------------"

# Run using mpirun
echo "Running MPI program..."
mpirun -n 1 ./runner