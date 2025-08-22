# PINNS Implementation on 3rd Order Non-Linear ODEs

🌐 **Overview**

This project utilizes a Physics-Informed Neural Network (PINN) to solve a 3rd order non-linear fluid-thermal system. The PINN is a hybrid approach that integrates the governing differential equations directly into the training process, alongside a small set of analytical data points, to find a robust and physically consistent solution.

🚀 **Run on Google Colab**

You can run this project instantly in a Google Colab environment by clicking the button below. This is the fastest way to get started without any local setup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xf_1aiV6FIUxoie4E26Ka64fFDvJCrH_?usp=sharing)

---
## 🚀 Getting Started

To get this project up and running, follow these simple steps.

1.  **Clone the Repository**
    Clone this repository to your local machine using the following command:
    ```bash
    git clone https://github.com/faizanfaiz11422/PINNverse.git
    cd 3rd-Order-NonLinear-System
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
    python NonLinear_System_PINNS.py
    ```
    *Alternatively, you can click the "Open In Colab" button above to run the code in a Google Colab notebook.*

## 💻 **Loss Function & Physics**

The system is defined by two coupled non-linear ordinary differential equations (ODEs) that describe the fluid flow and heat transfer. The PINN is trained to find the stream function $f(\eta)$ and the temperature profile $\theta(\eta)$ that satisfy these equations:

1. **Fluid Flow Equation ($R_1$)**

   $$f''' + (f')^2 - ff'' - De((f')^2f''' - 2ff'f'') + Df' = 0$$

2. **Thermal Equation ($R_2$)**

   $$\frac{1+R}{Pr} \theta'' + \frac{\theta_9}{\theta_{10}} f' \theta + S(\frac{\theta_6 \theta_9}{\theta_1 \theta_{10}} - \frac{\theta_5}{\theta_1}) \theta - f\theta' = 0$$

---
This function calculates the physics, boundary condition, and data loss components.

```python
@tf.function
def compute_loss(eta, analytic_eta, analytic_theta):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(eta)
        f, theta = model(eta)
        f1 = tape.gradient(f, eta)
        theta1_grad = tape.gradient(theta, eta)
        f2 = tape.gradient(f1, eta)
        theta2_grad = tape.gradient(theta1_grad, eta)
        f3 = tape.gradient(f2, eta)
    del tape

    # Physics residuals
    R1 = f3 + f1**2 - f*f2 - De*(f1**2*f3 - 2*f*f1*f2) + D*f1
    R2 = ((1 + R) / Pr) * theta2_grad + (theta9 / theta10) * f1 * theta + S * ((theta6 * theta9) / (theta1 * theta10) - (theta5 / theta1)) * theta - f * theta1_grad
    physics_loss = tf.reduce_mean(tf.square(R1)) + tf.reduce_mean(tf.square(R2))

    # Boundary conditions loss
    # ... (BC calculation omitted for brevity)

    # Data loss
    _, theta_pred_at_analytic = model(analytic_eta)
    data_loss = tf.reduce_mean(tf.square(theta_pred_at_analytic - analytic_theta))

    return physics_loss, bc_loss, data_loss, theta_pred_at_analytic

```

---

### ⚙️ **Physical Parameters**

The following physical parameters are used in the model:

| Parameter | Symbol | Value | Description |
 | ----- | ----- | ----- | ----- |
| Deborah Number | De | 4.0 | A measure of a material's fluidity. |
| Constant | D | 1.5 | A constant related to fluid properties. |
| Thermal Radiation | R | 1.0 | Represents heat transfer by radiation. |
| Heat Source | S | 3.3 | Represents an internal heat source/sink. |
| Prandtl Number | Pr | 1.0 | Ratio of momentum diffusivity to thermal diffusivity. |
| Constant | $\theta_1$ | 10.0 | System-specific constant. |
| Constant | $\theta_5$ | 10.0 | System-specific constant. |
| Constant | $\theta_6$ | 1.5 | System-specific constant. |
| Constant | $\theta_9$ | 2.0 | System-specific constant. |
| Constant | $\theta_{10}$ | 10.0 | System-specific constant. |
| Data Weighting | $\lambda_{data}$ | 10.0 | Weight for the analytical data loss. |

---

### 🚧 **Boundary Conditions**

The system is constrained by the following boundary conditions at $\eta=0$ and $\eta=1$:

* $f(0)=0$

* $f'(0)=1$

* $\theta(0)=1$

* $f(1)=0$

* $f''(1)=0$

* $\theta'(1)=0$

---

### 🧠 **Model & Training**

The neural network is composed of a simple, fully connected architecture:

| Layer (type) | Output Shape | Param # | Connected to |
|:---|:---|:---|:---|
| eta (InputLayer) | (None, 1) | 0 | - |
| dense (Dense) | (None, 32) | 64 | eta[0][0] |
| dense_1 (Dense) | (None, 64) | 2,112 | dense[0][0] |
| dense_2 (Dense) | (None, 128) | 8,320 | dense_1[0][0] |
| f (Dense) | (None, 1) | 129 | dense_2[0][0] |
| theta (Dense) | (None, 1) | 129 | dense_2[0][0] |

The model is trained for 10,000 epochs using the Adam optimizer. The total loss is a weighted sum of three components: physics loss, boundary condition loss, and data loss.

```python
# Training loop
for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        physics_loss, bc_loss, data_loss, theta_pred_at_analytic = compute_loss(train_eta, analytic_eta, analytic_theta)
        total_loss = physics_loss + bc_loss + lambda_data * data_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # ... plotting and printing logic

```

---

## 📊 **Results**

The final plots compare the PINN-predicted solutions with the analytical data.

![Alt text](https://github.com/faizanfaiz11422/PINNverse/blob/776e498deba4a77f05abd7a48e8ca4c7ef87f241/3rd-Order-NonLinear-System/Result.png)
