#!/bin/bash

# Usage:
# ./convert_all_mtx_to_bin.sh ./matrices ./binaries

INPUT_DIR=$1
OUTPUT_DIR=$2
EXECUTABLE=./cmake-build-debug/mtx_to_petsc_bin

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory $INPUT_DIR does not exist."
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

if [ ! -f "$EXECUTABLE" ]; then
  echo "Executable $EXECUTABLE not found. Please compile mtx_to_petsc_bin first."
  exit 1
fi

for file in "$INPUT_DIR"/*.mtx; do
  if [ -f "$file" ]; then
    echo "Converting $file ..."
    mpiexec -n 1 "$EXECUTABLE" "$file" "$OUTPUT_DIR"
  fi
done

echo "Conversion complete. Binary files saved in $OUTPUT_DIR"
