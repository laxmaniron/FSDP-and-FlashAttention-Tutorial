# PyTorch FSDP (Fully Sharded Data Parallel) Project

This project utilizes PyTorch's FSDP for training large models across multiple GPUs. Ensure you have at least PyTorch version 2.0 installed alongside the necessary dependencies.

## Prerequisites

- Python 3.x
- pip3
- A machine with atleast 2 GPUS
- CUDA 11.8

## Installation

1. **Install PyTorch, torchvision, and torchaudio:**
   Ensure you have at least PyTorch version 2.0. Run the following command to install PyTorch and its related libraries:
    ```bash
    pip3 install torch torchvision torchaudio
    ```

2. **Install PyTorch Profiler and TensorBoard:**
    - PyTorch Profiler is a tool to help you optimize your models. Install it using pip:
    ```bash
    pip3 install torch_tb_profiler
    ```
    - TensorBoard is a toolkit for visualizing machine learning experiments. Install it using pip:
    ```bash
    pip3 install tensorboard
    ```

3. **Validate the Installation:**
    You can check the installed versions of the libraries with the following Python commands:
    ```python
    import torch
    import torchvision
    import torchaudio
    print(torch.__version__)
    print(torchvision.__version__)
    print(torchaudio.__version__)
    ```

## Running the FSDP Training

1. **Navigate to your project directory:**
    ```bash
    cd /path/to/your/project
    ```

2. **Execute your PyTorch script with FSDP:**
    ```bash
    python3 FSDP_VIT.py
    ```

## Profiling and Visualization

1. **Profile your training script:**
    Add the necessary profiler code to your PyTorch script. Refer to the [PyTorch Profiler documentation](https://pytorch.org/docs/stable/profiler.html) for more details.

2. **Visualize the Profiler results with TensorBoard:**
    Launch TensorBoard and point it to the log directory created by PyTorch Profiler:
    ```bash
    tensorboard --logdir logs/mnist_profile/

3. Open a web browser and navigate to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard and analyze the results.

## Additional Resources

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/_modules/torch/distributed/fsdp.html)
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## Troubleshooting

If you encounter any issues or have questions, refer to the official documentation or contact the repository maintainers.
