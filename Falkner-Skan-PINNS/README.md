# Falkner-Skan PINN

A Physics-Informed Neural Network (PINN) designed to solve the Falkner-Skan boundary-layer equations. This project serves as a comprehensive example of how to combine deep learning with fundamental fluid dynamics and heat transfer principles.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kDYqmtj8o0sYbS_xzuM3eNzSXGVqAgN3?usp=sharing)

## 🚀 Getting Started

To get this project up and running, follow these simple steps.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/faizanfaiz11422/PINNverse.git](https://github.com/faizanfaiz11422/PINNverse.git)
    cd PINNverse/Falkner-Skan-PINNS
    ```

2.  **Install Dependencies**
    You will need a Python environment with TensorFlow, NumPy, and Matplotlib. It is recommended to use a virtual environment.
    ```bash
    pip install tensorflow numpy pandas matplotlib
    ```

3.  **Add Your Data**
    Place your `SampleData.csv` file directly into the repository's root folder. This file contains the analytic solution used to guide the PINN's training.

4.  **Run the Code**
    Execute the main Python script from your terminal.
    ```bash
    python FS_PINNS.py
    ```
    *Alternatively, you can click the "Open In Colab" button above to run the code in a Google Colab notebook.*

## 🧠 How It Works

This project uses a hybrid PINN approach that combines three loss components to train the neural network:

1.  **Physics Loss**: This is the core of the PINN. It ensures the model's output satisfies the governing differential equations of the Falkner-Skan problem.
2.  **Boundary Condition (BC) Loss**: This component enforces the known conditions at the edges of the domain (e.g., at the wall and far away from the wall).
3.  **Supervised Data Loss**: This is where your analytic data comes in. It compares the model's prediction directly with your `SampleData.csv` file, helping to guide the PINN toward the correct solution and improve accuracy.

The training process minimizes a weighted sum of these three losses, resulting in a model that is both physically consistent and accurate.

## 📊 Sample Results

This project generates plots to visualize the model's performance and predictions. The following image is an example of the output you can expect after a successful training run.

![Temperature Profile Comparison](https://github.com/faizanfaiz11422/PINNverse/blob/53f86d9e16b588476dd1ee29552adbd837f273ee/Falkner-Skan-PINNS/Result.png)

*This plot compares the PINN's temperature profile (`θ(η)`) prediction (blue line) against the analytic solution from your data (red dashed line).*
