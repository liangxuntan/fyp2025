#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=400G
#SBATCH --partition=large_336
#SBATCH --output=prot2pfam_%j.out

source /home/software/anaconda3/bin/activate hmmer_env

echo "Detected CPUs: $(nproc)"
lscpu | grep "^CPU(s):"
echo "Total memory (GB): $(free -g | awk '/^Mem:/ {print $2}')"
free -h

#Start resource logger
(
  while true; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    echo "[$timestamp] ===== Resource usage ====="
    top -b -n 1 | head -n 12
    echo ""
    sleep 1200
  done
) >> /home/liangxun/logs/prot2pfam_resource_usage_${SLURM_JOB_ID:-$$}.log 2>&1 &
LOGGER_PID=$!

cd /home/liangxun/scripts
./prot2pfam_cluster.sh -c 1 -j 250 -e 1e-3 /home/liangxun/prodigal_output/*

kill $LOGGER_PID

# echo "SLURM usage for this job:"
# sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode

conda deactivate