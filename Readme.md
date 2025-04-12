# Laplacian-Eigenfunction Based Neural Operator

This repository contains code for implementing the Laplacian-Eigenfunction Based Neural Operator (LENO).

## Overview
The workflow consists of three main steps:

1. **Generate Initial Data**
   - Use MATLAB scripts `gen_initial_1d.m` or `gen_initial_2d.m` to generate 1D or 2D initial data under various boundary conditions.

2. **Generate Dynamic Solutions**
   - Use Python scripts (`FEM_xx.py`) to compute the corresponding dynamic solutions using Finite Element Methods (FEM).

3. **Run Experiments**
   - Use Python scripts (`LENO_xx.py`) to train and test the Laplacian-Eigenfunction Based Neural Operator on the generated data.

## Prerequisites
### MATLAB:
- Ensure MATLAB is installed for running the initial data generation scripts.

### Python:
- This code depends on PyTorch.

## Steps to Execute

### Step 1: Generate Initial Data
Run the MATLAB scripts to create the initial data:
- For 1D data:
  ```
  gen_initial_1d.m
  ```
- For 2D data:
  ```
  gen_initial_2d.m
  ```
  Customize the boundary conditions in these scripts as needed.

### Step 2: Compute Dynamic Solutions
Use the FEM Python scripts to compute dynamic solutions based on the generated initial data:
- Example command:
  ```bash
  python FEM_xx.py
  ```

### Step 3: Run LENO Experiments
Train and test the neural operator using the LENO scripts:
- Example command:
  ```bash
  python LENO_xx.py
  ```

## Notes
- Replace `xx` in script names with the appropriate configuration (e.g., ac (allen-cahn), kpp (Fisher-KPP)).
