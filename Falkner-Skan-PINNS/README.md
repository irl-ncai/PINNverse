# PINNverse

A collection of Physics-Informed Neural Networks (PINNs) for solving engineering and science problems. This repository demonstrates how to combine neural networks with fundamental physics to create robust, data-informed models for various applications.

This project's initial focus is on a detailed example of a PINN solving the **Falkner-Skan equations**, a classical boundary-layer problem in fluid dynamics.

## 🚀 Getting Started

To get this project up and running, follow these simple steps.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/PINNverse.git](https://github.com/your-username/PINNverse.git)
    cd PINNverse
    ```

2.  **Install Dependencies**
    You will need a Python environment with TensorFlow, NumPy, and Matplotlib. It is recommended to use a virtual environment.
    ```bash
    pip install tensorflow numpy pandas matplotlib
    ```

3.  **Add Your Data**
    Place your `mydata1.csv` file directly into the repository's root folder. This file contains the analytic solution used to guide the PINN's training.

4.  **Run the Code**
    Execute the main Python script from your terminal.
    ```bash
    python pinn_falkner_skan.py
    ```

## 🧠 How It Works

This project uses a hybrid PINN approach that combines three loss components to train the neural network:

1.  **Physics Loss**: This is the core of the PINN. It ensures the model's output satisfies the governing differential equations of the Falkner-Skan problem.
2.  **Boundary Condition (BC) Loss**: This component enforces the known conditions at the edges of the domain (e.g., at the wall and far away from the wall).
3.  **Supervised Data Loss**: This is where your analytic data comes in. It compares the model's prediction directly with your `mydata1.csv` file, helping to guide the PINN toward the correct solution and improve accuracy.

The training process minimizes a weighted sum of these three losses, resulting in a model that is both physically consistent and accurate.

## 📈 Results

After training, the script will output three plots:

1.  A comparison of the PINN's temperature profile (`θ(η)`) against the analytic solution from your CSV file.
2.  The training loss history, which shows how the model's error decreases over time.
3.  A plot of the PINN's predicted stream function (`f(η)`).

A successful run will show the PINN's prediction curve closely matching the analytic data, demonstrating the power of the PINN approach.

## 💡 Future Projects

`PINNverse` is designed to be a growing collection. Future projects may include:

* Solving other fluid dynamics problems (e.g., Couette flow).
* Applying PINNs to heat transfer or structural mechanics.
* Exploring different neural network architectures and activation functions.

Feel free to contribute by adding your own PINN projects!

