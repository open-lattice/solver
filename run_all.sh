#!/bin/bash

EXEC=./cmake-build-debug/main
MTX_DIR="./matrices"
OUTCSV="timing_log.csv"

CONSTRAINT_COUNTS=(100 500 1000 2000 5000 10000 50000 100000 250000 500000 1000000 5000000)

PROCS=(1 2 3 4 5 6)

declare -A BASELINES

# header
echo "matrix,rows,cols,nnz,sparsity,constraints,total_time_s,constraint_time_s,procs,speedup,efficiency" > "$OUTCSV"

for mtx in "$MTX_DIR"/*.mtx; do
    matrix=$(basename "$mtx")
    matrix_rows=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')

    for nC in "${CONSTRAINT_COUNTS[@]}"; do
        if [ "$nC" -le "$matrix_rows" ]; then
            for np in "${PROCS[@]}"; do
                echo "Running: $matrix with $nC constraints on $np processes"
                mpiexec -n "$np" "$EXEC" "$mtx" "$nC"

                line=$(tail -n 1 "$OUTCSV" | cut -d',' -f1-8)
                constraint_time=$(echo "$line" | awk -F',' '{print $(NF)}')

                key="$matrix:$nC"
                if [ "$np" -eq 1 ]; then
                    BASELINES["$key"]=$constraint_time
                    speedup=1
                    efficiency=1
                else
                    baseline=${BASELINES["$key"]}
                    speedup=$(awk "BEGIN {printf \"%.6f\", $baseline / $constraint_time}")
                    efficiency=$(awk "BEGIN {printf \"%.6f\", $speedup / $np}")
                fi

                sed -i '$ d' "$OUTCSV"

                echo "$line,$np,$speedup,$efficiency" >> "$OUTCSV"
            done
        else
            echo "Skipping: $matrix with $nC constraints (exceeds $matrix_rows rows)"
        fi
    done
done

echo "Benchmark complete. Results saved in $OUTCSV"
