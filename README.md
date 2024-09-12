Mandelbrot Set Visualization with OpenCL

This project generates high-resolution images of the Mandelbrot set using OpenCL for parallel computation on the GPU. The images are created with a zoom effect to explore deeper details of the fractal, utilizing double precision for accurate calculations.
Features

    High-Resolution Rendering: Images are generated with a resolution of 2048x2048 pixels.
    Double Precision Computation: Ensures high precision when zooming into detailed parts of the Mandelbrot set.
    OpenCL Acceleration: Uses GPU acceleration for fast computation of the Mandelbrot set.
    Multi-Threaded Image Saving: Uses a worker thread to save generated images without blocking the computation.
    Customizable Parameters: Easily adjust the number of images, resolution, and zoom factor.

Prerequisites

    Python 3.10 or higher
    NumPy for numerical computations
    Matplotlib for image generation and saving
    PyOpenCL for OpenCL integration
    An OpenCL-compatible GPU

    Installation

    1. Install the required Python libraries:

    pip install numpy matplotlib pyopencl

Usage

Run the script to generate Mandelbrot set images:

python mandelbrot_opencl.py


Adjustable Parameters

    width and height: Dimensions of the output image.
    max_iterations: Maximum number of iterations for the Mandelbrot computation.
    num_images: Number of images to generate.
    zoom_factor: Factor by which the zoom increases with each image.

Output

Generated images will be saved in the mandelbrot_images directory.
Example Output

How It Works

    OpenCL Kernel: Computes the Mandelbrot set for each pixel using double precision.
    Image Generation: The script generates a sequence of images, zooming into a specific region of the Mandelbrot set.
    Multi-Threaded Saving: A separate thread handles saving images to disk, ensuring smooth and continuous computation.

Troubleshooting

    OpenCL Errors: Ensure your GPU drivers are up-to-date and that OpenCL is correctly installed.
    Memory Overflow: If you encounter memory issues, reduce the num_images or the image resolution.

Contributing

Feel free to fork the repository and submit pull requests. Contributions are welcome!
License

This project is licensed under the MIT License.