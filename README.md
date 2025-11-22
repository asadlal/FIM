# Father-Inspired Metaheuristic (FIM)

This repository contains the MATLAB implementation of the Father-Inspired Metaheuristic (FIM), all competitor algorithms, and the complete set of results used in the paper.

## Repository Structure

### Matlab Code

This folder includes all files required to run the experiments.

#### Algorithms
Contains FIM and all competitor algorithms (GA, TLBO, WSO, WO, GSA, WOA, GWO, AVOA, RSA, MPA, TSA, MVO).

#### Main Files
- `FIM_Comparison.m` - The main script. It runs FIM and all competitor algorithms on the selected benchmarks and automatically saves all results such as Excel files, convergence plots and representative graphs.
- `run_ranksum.m` - Runs the Wilcoxon rank-sum test and stores the p-value results.
- `CEC2017.m` - Defines the CEC2017 benchmark functions (F1 to F30). Function F2 is excluded.
- `fun_info.m` - Defines the classical benchmark functions F1 to F23.
- `Func_eng.m` - Defines the engineering design problems: Pressure Vessel, Spring Design, Three-Bar Truss, Gear Train and Cantilever Beam.
- `input_data` - Contains additional files required by some benchmark problems.

## How to Run the Experiments

1. Open MATLAB.
2. Set the folder "Matlab Code" as your working directory.
3. Run the following command:
   ```matlab
   FIM_Comparison
