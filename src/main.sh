#!/bin/bash

# Define the number of iterations per condition
ITERATIONS=1000

# Define the sample sizes
SAMPLE_SIZES=(10 40 4000)

# Run the simulation for each sample size
for N in "${SAMPLE_SIZES[@]}"; do
    echo "Running simulation for N=$N with $ITERATIONS iterations..."
    python src/simulate.py $N $ITERATIONS
    echo "Simulation for N=$N completed."
done

