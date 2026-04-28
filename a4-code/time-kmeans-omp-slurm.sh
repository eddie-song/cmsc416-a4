#!/bin/bash -l

## BASIC JOB SETUP
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=250m
#SBATCH --account cmsc416-class

## SELECT PARTITION
#SBATCH --partition=standard
# #SBATCH --partition=debug

## SET OUTPUT BASED ON SCRIPT NAME
#SBATCH --output=%x.job-%j.out
#SBATCH --error=%x.job-%j.out

# # enable email notification of job completion
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=profk@umd.edu

# ADJUST: location of project code with respect to home directory
cd ~/a4-code

# function to time a command but suppress the time if it fails due to
# errors; avoids reporting times on error runs that students often
# commit and don't check
# usage: dotime "some command to run" "format string for time command"
function dotime () {
    cmd="$1"
    timestring="$2"
    printf '>> %s\n' "$cmd"
    out=$(/usr/bin/time -f "$timestring" $cmd 2>&1)
    exit_code=$?
    if [[ "$exit_code" == "0" ]]; then
        printf "%b\n" "$out"
    else
        printf "%b\n" "$out" |grep -v runtime
        echo "runtime: FAILED with exit code $exit_code"
    fi
    echo
}

# Fixed parameters for all runs of kmeans
DATADIR="sample-mnist-data"
NCLUST=20
MAXITERS=500
mkdir -p outdirs   # subdir or all output kmeans output

# Full performance benchmark of all combinations of data files and
# processor counts
ALLDATA="digits_all_5e3.txt digits_all_1e4.txt digits_all_3e4.txt"
ALLNP="1 4 8 16 36 64 85 128"

# # Small sizes for testing
# ALLDATA="digits_all_5e3.txt"
# ALLNP="1 4 16"

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for DATA in $ALLDATA; do
        OUTDIR=outdirs/outdir_${DATA}_${NP}
        export OMP_NUM_THREADS=$NP
        cmd="./kmeans_omp $DATADIR/$DATA $NCLUST $OUTDIR $MAXITERS"
        timefmt="runtime: procs $NP data $DATA realtime %e"
        dotime "$cmd" "$timefmt"
        echo
    done
done
