GPU Vector Operations using CUDA
Experiment Overview
This repository contains the CUDA vector-operation experiment, empirical benchmark results, performance visualizations, correctness verification, and performance analysis for:
- CPU Sequential Execution: Vector addition and element-wise vector multiplication executed sequentially on the CPU
- CUDA GPU Execution: Vector addition and element-wise vector multiplication executed in parallel on an NVIDIA GPU
The experiment evaluates different vector sizes and different CUDA threads-per-block configurations.
Operations
- Vector Addition
- Element-wise Vector Multiplication
Benchmark Measurements
- CPU addition time
- CPU multiplication time
- GPU addition kernel time
- GPU multiplication kernel time
- Total GPU phase time
- Addition kernel speedup
- Multiplication kernel speedup
- End-to-end speedup
- Correctness verification
Standard Experimental Configuration
Parameter	Configuration
GPU	NVIDIA RTX 4500 Ada Generation
NVIDIA Driver	610.47
CUDA Toolkit	13.4
NVCC	13.4.92
Host Compiler	Microsoft Visual C++
Vector Sizes	1,000; 10,000; 100,000; 1,000,000; 10,000,000
Threads per Block – Vector Study	256
Vector Size – Thread Study	10,000,000
Threads per Block – Thread Study	128; 256; 512


CUDA Programming Model
CUDA organizes GPU execution using a hierarchy of:
- Grid
- Blocks
- Threads
The global thread index is calculated using:
int i = blockIdx.x * blockDim.x + threadIdx.x;
The kernel uses a boundary check:
if (i < N)
{
    C[i] = A[i] + B[i];
}
The same indexing approach is used for element-wise multiplication.
The number of blocks required for a vector is calculated using:
int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
This ensures that all vector elements are covered.
Vector Operations
Vector Addition
The vector addition operation is:
C[i] = A[i] + B[i]
Example:
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]

C = [6, 8, 10, 12]
Element-wise Vector Multiplication
The vector multiplication operation is:
C[i] = A[i] * B[i]
Example:
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]

C = [5, 12, 21, 32]
CPU vs GPU Execution
CPU Sequential Execution
The CPU processes the vector elements sequentially:
for (int i = 0; i < N; i++)
{
    C[i] = A[i] + B[i];
}
For multiplication:
for (int i = 0; i < N; i++)
{
    C[i] = A[i] * B[i];
}
CUDA GPU Execution
The GPU assigns different vector elements to different CUDA threads:
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N)
{
    C[i] = A[i] + B[i];
}
This allows independent vector elements to be processed in parallel.
Lab Manual: Execution Steps & Commands
Part A: CUDA Environment Verification
1. Verify NVIDIA GPU
Open PowerShell and run:
nvidia-smi
This verifies the installed NVIDIA GPU, driver information, GPU memory and CUDA-related information.
2. Verify CUDA Toolkit
nvcc --version
Recorded CUDA compiler:
CUDA compilation tools, release 13.4
V13.4.92
3. Locate CUDA Compiler
where.exe nvcc
This displays the path of the installed nvcc compiler.
4. Locate Host Compiler
where.exe cl
This verifies that the Microsoft C/C++ host compiler is available.
Part B: Vector Operation Benchmark
1. Start the Benchmark Program
Run:
.\vector_operations.exe
The program accepts the vector size and threads-per-block configuration.
2. Vector-Size Study
Run the benchmark for:
N = 1,000
N = 10,000
N = 100,000
N = 1,000,000
N = 10,000,000
Use:
Threads per Block = 256
For each configuration, record:
- CPU addition time
- CPU multiplication time
- GPU addition kernel time
- GPU multiplication kernel time
- Total GPU phase time
- Addition speedup
- Multiplication speedup
- End-to-end speedup
- Correctness
3. Thread Configuration Study
Use:
Vector Size = 10,000,000
Run the experiment using:
128 threads/block
256 threads/block
512 threads/block
For each configuration, record GPU kernel timing, total GPU time, speedup and correctness.
Part C: Timing
GPU kernel timing is performed using CUDA events:
cudaEventRecord(start);

kernel<<<blocks, threadsPerBlock>>>(...);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&milliseconds, start, stop);
CPU execution is timed separately.
Part D: Graph Generation
Install the required Python packages:
pip install pandas matplotlib
Run:
python scripts\plot_results.py
The generated graphs are stored in the graphs/ directory.
Performance Formulas
Kernel Speedup
Kernel Speedup = CPU Time / GPU Kernel Time
End-to-End Speedup
End-to-End Speedup = CPU Time / Total GPU Phase Time
A speedup greater than 1× indicates that the GPU measurement is faster.
A speedup below 1× indicates that the CPU measurement is faster.
Results
Vector Size Benchmark Results
The following table contains the recorded average measurements using 256 threads per block.
N	CPU Add (ms)	CPU Mul (ms)	GPU Add Kernel (ms)	GPU Mul Kernel (ms)	Total GPU (ms)	Add Speedup	Mul Speedup	End-to-End
1,000	0.000200	0.000200	0.021352	0.030736	0.143656	0.011344×	0.008888×	0.003478×
10,000	0.001700	0.001300	0.021504	0.034475	0.168395	0.081705×	0.051797×	0.020456×
100,000	0.017733	0.017833	0.027893	0.013344	0.281461	0.667200×	2.053424×	0.129574×
1,000,000	0.261700	0.244600	0.027029	0.010240	1.343179	9.672405×	23.886719×	0.379052×
10,000,000	3.499700	3.291460	0.327718	0.331162	11.802694	10.734719×	9.939625×	0.577858×


Thread Configuration Benchmark Results
For N = 10,000,000:
Threads per Block	GPU Add Kernel (ms)	GPU Mul Kernel (ms)	Total GPU (ms)	End-to-End Speedup
128	0.349232	0.329728	12.103680	0.551406×
256	0.327718	0.331162	11.802694	0.577858×
512	0.316000	0.330752	10.944885	0.630326×


Correctness Verification
The recorded experiment executions were checked for both vector operations.
Operation	PASS	FAIL
Vector Addition	23	0
Vector Multiplication	23	0


All 23 recorded executions passed correctness verification for both operations.
Performance Graphs
1. Execution Time vs Vector Size

The graph compares CPU execution time and GPU execution time as vector size increases.
For small vector sizes, the GPU does not provide an overall performance advantage because GPU-related overhead is large compared with the amount of computation.
As the vector size increases, CPU execution time increases more significantly while GPU kernel execution remains comparatively low.
This demonstrates that GPU parallelism becomes more useful as the amount of independent work increases.
2. Speedup vs Vector Size

The speedup graph shows the change in GPU kernel performance with vector size.
For small vector sizes, the GPU kernel speedup is below 1×.
As the vector size increases, the GPU receives more independent work and the kernel becomes significantly faster relative to the CPU.
The highest recorded multiplication kernel speedup is:
23.886719×
at:
N = 1,000,000
The highest recorded addition kernel speedup is:
10.734719×
at:
N = 10,000,000
3. GPU Kernel Time vs Threads per Block

The thread configuration experiment uses:
N = 10,000,000
The tested configurations are:
128 threads/block
256 threads/block
512 threads/block
The lowest average total GPU time among the tested configurations is:
10.944885 ms
for:
512 threads/block
Performance Analysis
1. Effect of Vector Size
For small vectors, the GPU overhead dominates because the computation is very small.
At larger vector sizes, more elements can be processed in parallel, making GPU kernel execution significantly faster.
The recorded results demonstrate a strong increase in kernel speedup as the vector size increases.
2. Effect of Threads per Block
The thread configuration experiment shows that changing the number of threads per block affects GPU performance.
Threads per Block	Total GPU Time
128	12.103680 ms
256	11.802694 ms
512	10.944885 ms


Among the tested configurations, 512 threads per block gives the lowest total GPU time.
3. Kernel Speedup vs End-to-End Speedup
A key observation is:
Kernel Speedup ≠ End-to-End Speedup
The GPU kernels can be significantly faster than the CPU operations, while the complete GPU phase can still be slower.
This is because the complete GPU execution includes operations such as:
Host → Device Transfer
        ↓
Kernel Launch
        ↓
GPU Kernel Execution
        ↓
Device → Host Transfer
        ↓
Synchronization
For simple vector operations, the computation per element is small. Therefore, memory-transfer and execution overhead can dominate the complete execution time.
4. Best Recorded Kernel Speedups
Operation	Vector Size	Highest Recorded Kernel Speedup
Vector Addition	10,000,000	10.734719×
Vector Multiplication	1,000,000	23.886719×


Key Findings
1. GPU execution is not automatically faster for every vector size.
2. Small workloads are strongly affected by GPU overhead.
3. Larger vector sizes provide more parallel work for the GPU.
4. GPU kernel speedup becomes significant for larger vector sizes.
5. The highest recorded multiplication kernel speedup is 23.886719×.
6. The highest recorded addition kernel speedup is 10.734719×.
7. End-to-end speedup remained below 1× for the recorded vector-size configurations.
8. The 512-thread configuration produced the lowest average total GPU time among the tested thread configurations.
9. All recorded executions passed correctness verification.
10. Kernel-level performance and end-to-end application performance must be analyzed separately.
Repository Directory Structure
.
├── .gitignore
├── README.md
│
├── docs/
│   └── CUDA_Vector_Operations_Observations_Analysis_Conclusion.docx
│
├── graphs/
│   ├── graph1_time_vs_size.png
│   ├── graph2_speedup_vs_size.png
│   └── graph3_kernel_vs_threads.png
│
├── results/
│   ├── results.csv
│   └── summary.csv
│
├── scripts/
│   └── plot_results.py
│
├── screenshots/
│   ├── Screenshot 2026-09-18 102643.png
│   ├── Screenshot 2026-10-06 104814.png
│   ├── Screenshot 2026-10-06 104830.png
│   ├── Screenshot 2026-10-06 104858.png
│   ├── Screenshot 2026-10-06 105019.png
│   ├── Screenshot 2026-10-06 105100.png
│   ├── Screenshot 2026-10-06 110326.png
│   ├── Screenshot 2026-10-06 110345.png
│   ├── Screenshot 2026-10-06 110432.png
│   ├── Screenshot 2026-10-06 110528.png
│   ├── Screenshot 2026-10-06 110558.png
│   ├── Screenshot 2026-10-06 110715.png
│   ├── Screenshot 2026-10-06 110743.png
│   ├── Screenshot 2026-10-06 110810.png
│   ├── Screenshot 2026-10-06 110839.png
│   ├── Screenshot 2026-10-06 110946.png
│   ├── Screenshot 2026-10-06 111125.png
│   ├── Screenshot 2026-10-06 111152.png
│   └── Screenshot 2026-10-06 111216.png
│
└── src/
    ├── cuda_test.cu
    └── cuda_test.cu.txt
Result Files
results/results.csv
Contains the individual recorded benchmark measurements.
results/summary.csv
Contains the averaged measurements used for the final tables and graphs.
graphs/
Contains the three generated performance graphs.
screenshots/
Contains the experimental evidence collected during environment verification and benchmark execution.
docs/
Contains the detailed observations, analysis and conclusion document.
Quick Command Reference
GPU Verification
nvidia-smi
CUDA Verification
nvcc --version
Locate NVCC
where.exe nvcc
Locate Host Compiler
where.exe cl
Run Benchmark
.\vector_operations.exe
Install Graph Dependencies
pip install pandas matplotlib
Generate Graphs
python scripts\plot_results.py
Final Performance Summary
Metric	Result
GPU	NVIDIA RTX 4500 Ada Generation
CUDA Toolkit	13.4
NVCC	13.4.92
Largest Vector Size	10,000,000
Vector-Size Study Threads	256
Thread Study Vector Size	10,000,000
Highest Addition Kernel Speedup	10.734719×
Highest Multiplication Kernel Speedup	23.886719×
Best Tested Threads/Block	512
Lowest Total GPU Time	10.944885 ms
Addition Correctness	23 PASS / 0 FAIL
Multiplication Correctness	23 PASS / 0 FAIL


Conclusion
The CUDA Vector Operations experiment demonstrates the use of NVIDIA CUDA to parallelize vector addition and element-wise vector multiplication.
The results show that GPU acceleration depends strongly on workload size. For small vectors, GPU overhead can dominate the computation, making CPU execution faster.
As the vector size increases, the GPU can process a larger number of independent elements in parallel. This produces significant kernel-level speedups.
The highest recorded multiplication kernel speedup was 23.886719×, while the highest recorded addition kernel speedup was 10.734719×.
However, the complete GPU phase remained slower than the CPU baseline in the recorded vector-size experiments because the total GPU measurement includes additional overhead such as memory transfers and synchronization.
The thread configuration study showed that 512 threads/block produced the lowest average total GPU time among the tested configurations.
The main conclusion is that GPU kernel speedup and end-to-end application speedup are different performance measures. For simple memory-oriented vector operations, data movement and execution overhead can dominate total execution time even when the GPU kernels themselves are significantly faster.
Author
Renuka Kagadal
