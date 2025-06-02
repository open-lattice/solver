#!/bin/bash

EXEC=./cmake-build-debug/main
MTX_DIR="./binaries"
OUTCSV="timing_log.csv"

CONSTRAINT_COUNTS=(100 500 1000 2000 5000 10000 50000 100000 250000 500000 1000000)
PROCS=(1 2 3 4 5 6)

declare -A BASELINES

# Header
echo "matrix,rows,cols,nnz,sparsity,constraints,total_time_s,constraint_time_s,procs,speedup,efficiency,format" > "$OUTCSV"

for matrix_file in "$MTX_DIR"/*; do
    filename=$(basename "$matrix_file")
    extension="${filename##*.}"
    name="${filename%.*}"

    # Detect matrix size
    if [[ "$extension" == "mtx" ]]; then
        matrix_rows=$(grep -v '^%' "$matrix_file" | head -1 | awk '{print $1}')
        matrix_cols=$(grep -v '^%' "$matrix_file" | head -1 | awk '{print $2}')
        matrix_nnz=$(grep -v '^%' "$matrix_file" | head -1 | awk '{print $3}')
    elif [[ "$extension" == "bin" ]]; then
        info_file="${MTX_DIR}/${name}.info"
        if [[ -f "$info_file" ]]; then
            read matrix_rows matrix_cols matrix_nnz < "$info_file"
        else
            echo "Missing .info file for $filename. Skipping."
            continue
        fi
    else
        echo "Unsupported file type: $extension. Skipping."
        continue
    fi

    for nC in "${CONSTRAINT_COUNTS[@]}"; do
        if [ "$nC" -le "$matrix_rows" ]; then
            for np in "${PROCS[@]}"; do
                echo "Running: $filename with $nC constraints on $np processes"
                mpiexec -n "$np" "$EXEC" "$matrix_file" "$nC"

                line=$(tail -n 1 "$OUTCSV" | cut -d',' -f1-8)
                constraint_time=$(echo "$line" | awk -F',' '{print $(NF)}')

                key="$filename:$nC"
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
                sparsity=$(awk "BEGIN {printf \"%.6f\", 1.0 - ($matrix_nnz / ($matrix_rows * $matrix_cols))}")
                echo "$filename,$matrix_rows,$matrix_cols,$matrix_nnz,$sparsity,$nC,$line,$np,$speedup,$efficiency,$extension" >> "$OUTCSV"
            done
        else
            echo "Skipping: $filename with $nC constraints (exceeds $matrix_rows rows)"
        fi
    done
done

echo "Benchmark complete. Results saved in $OUTCSV"
