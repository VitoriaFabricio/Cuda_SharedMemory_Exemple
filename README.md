# Cuda_SharedMemory_Exemple

This repository contains a CUDA implementation for applying a 3x3 filter to an image using shared memory.

The main components include:

CUDA Kernel:

- Implements a parallel filter application using CUDA.
- Defines a 3x3 filter matrix (filter) and computes the sum of its elements.
- Handles edge and corners cases for pixels at the borders and corners of the image (input_image).
- Computes the weighted sum (sum) using the filter matrix and normalizes it by filter_sum.
- Writes the filtered output to output_image.
  
Host Code:

- Reads an image from a CSV file (original_0.csv).
- Initializes CUDA environment: platform, device, context, command queue.
- Reads and builds the CUDA kernel (filter.cu).
- Sets up memory buffers and transfers data between host and device.
- Executes the kernel to apply the filter to the image.
- Measures and prints the execution time.
- Reads the processed image from the device and writes the result to a CSV file (result.csv).

  
Compilation:

  ```nvcc kernel1.cu -o exit  ```

Execution:

  ```./exit  ```
