\# GPU Vector Operations using CUDA



\## Overview



This experiment implements vector addition and element-wise vector multiplication using sequential CPU execution and CUDA GPU parallelism.



The experiment compares CPU and GPU execution time, verifies correctness, evaluates speedup, and studies the effect of vector size and CUDA threads per block.



Each vector element is assigned to a CUDA thread for parallel processing.



\## Objectives



\- Implement vector addition using CPU.

\- Implement vector multiplication using CPU.

\- Implement vector addition using CUDA.

\- Implement vector multiplication using CUDA.

\- Verify CPU and GPU correctness.

\- Measure CPU and GPU execution time.

\- Calculate GPU speedup.

\- Compare different vector sizes.

\- Compare different threads-per-block configurations.

\- Generate performance graphs.



\## Technologies Used



\- C++

\- NVIDIA CUDA

\- CUDA `nvcc`

\- NVIDIA GPU

\- Python 3

\- pandas

\- matplotlib

\- Windows PowerShell



\## Project Structure



```text

gpu-vector-operations/

│

├── README.md

├── .gitignore

│

├── src/

│   └── cuda\_test.cu

│

├── scripts/

│   └── plot\_results.py

│

├── results/

│   ├── results.csv

│   └── summary.csv

│

├── graphs/

│   ├── graph1\_time\_vs\_size.png

│   ├── graph2\_speedup\_vs\_size.png

│   └── graph3\_kernel\_vs\_threads.png

│

├── screenshots/

│   └── experiment screenshots

│

└── docs/

&#x20;   ├── observations.md

&#x20;   ├── conclusion.md

&#x20;   └── experiment analysis document

