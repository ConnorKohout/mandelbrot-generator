import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering to a file

import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
from threading import Thread
from queue import Queue

# Parameters for the Mandelbrot set
width, height = 2048, 2048
max_iterations = 1000
num_images = 4000

# Define the OpenCL kernel for computing the Mandelbrot set using double precision
mandelbrot_kernel = """
__kernel void mandelbrot(__global int *output, const int width, const int height,
                         const double real_min, const double real_max,
                         const double imag_min, const double imag_max,
                         const int max_iterations) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = j * width + i;

    double real = real_min + i * (real_max - real_min) / width;
    double imag = imag_min + j * (imag_max - imag_min) / height;

    double z_real = real;
    double z_imag = imag;
    int iteration = 0;

    while (z_real * z_real + z_imag * z_imag < 4.0 && iteration < max_iterations) {
        double temp_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp_real;
        iteration++;
    }

    output[index] = iteration;
}
"""

# Create the OpenCL context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Create output buffer once and reuse
output = np.zeros((width, height), dtype=np.int32)  # Keep output as int32
output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Compile the OpenCL kernel with double precision support
program = cl.Program(context, mandelbrot_kernel).build(["-cl-fast-relaxed-math"])

# Create a directory to store images
output_dir = "mandelbrot_images"
os.makedirs(output_dir, exist_ok=True)

# Set initial parameters for the Mandelbrot computation
real_center, imag_center = -0.743643887037151, 0.131825904205330  # Seahorse Valley
zoom_factor = np.float64(1.02)  # Slower zoom for deeper exploration

# Create a queue for saving images with a maximum size
image_queue = Queue(maxsize=10)  # Limit the queue size to prevent memory overflow

def save_image_worker():
    while True:
        item = image_queue.get()
        if item is None:  # Stop the worker if None is received
            break

        scaled_output, real_min, real_max, imag_min, imag_max, i = item
        norm = Normalize(vmin=np.min(scaled_output), vmax=np.max(scaled_output))
        plt.imshow(scaled_output, extent=(real_min, real_max, imag_min, imag_max), cmap='inferno', norm=norm)
        plt.axis('off')
        plt.title(f'Mandelbrot Set - Zoom Level {zoom_factor ** i:.2f}')
        plt.savefig(f"{output_dir}/mandelbrot_{i:04d}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        image_queue.task_done()

# Start the worker thread
worker_thread = Thread(target=save_image_worker)
worker_thread.start()

# Generate images
for i in range(num_images):
    # Adjust zoom for each image to focus on detailed parts
    zoom = zoom_factor ** np.float64(i)
    
    # Calculate the new bounds for zooming using double precision
    real_range = np.float64(3.0) / zoom
    imag_range = np.float64(3.0) / zoom
    
    new_real_min = real_center - real_range / 2
    new_real_max = real_center + real_range / 2
    new_imag_min = imag_center - imag_range / 2
    new_imag_max = imag_center + imag_range / 2

    # Print bounds for debugging purposes
    print(f"Image {i}: Bounds - Real: ({new_real_min}, {new_real_max}), Imag: ({new_imag_min}, {new_imag_max})")

    # Asynchronous kernel execution
    event = program.mandelbrot(queue, (width, height), None, output_buffer,
                               np.int32(width), np.int32(height),
                               np.float64(new_real_min), np.float64(new_real_max),
                               np.float64(new_imag_min), np.float64(new_imag_max),
                               np.int32(max_iterations))
    event.wait()  # Ensure kernel execution is complete
    
    # Non-blocking read of buffer to allow overlapping of computation and data transfer
    cl.enqueue_copy(queue, output, output_buffer, wait_for=[event], is_blocking=False)
    
    # Convert to float for scaling
    scaled_output = np.log1p(output.astype(np.float32))  # Convert to float32 for log scaling

    # Queue the image data for saving by the worker thread
    image_queue.put((scaled_output, new_real_min, new_real_max, new_imag_min, new_imag_max, i))

    print(f"Generated image {i+1}/{num_images}")

# Wait for all images to be processed
image_queue.join()

# Stop the worker thread
image_queue.put(None)
worker_thread.join()

print("All images generated!")
