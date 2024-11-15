# minigrad
minigrad: A minimalistic autograd engine to demystify backpropagation and automatic differentiation. Built with simplicity in mind, it provides a clear and intuitive implementation of tensor operations, enabling step-by-step gradient computation for educational purposes.
Here’s the revised documentation for **Minigrad.py**, emphasizing clarity, alignment with the code, and accessibility:
---

## **Introduction**

Automatic differentiation (autograd) is at the heart of modern deep learning libraries. **Minigrad** replicates this functionality by:
- Supporting basic tensor operations like addition, multiplication, and summation.
- Tracking operations and building a computational graph for backpropagation.
- Calculating gradients for all tensor operations.

Minigrad simplifies the concept of autograd by focusing on clarity and essential functionality without unnecessary complexity.

---

## **Key Features**

1. **Custom Tensor Class**: Wraps NumPy arrays and tracks gradients for operations.
2. **Gradient Propagation**: Automatically computes gradients through the computational graph.
3. **Core Mathematical Operations**: Implements addition, multiplication, summation, and more, with gradient tracking.
4. **Educational Use**: Designed to be small, simple, and easy to understand, making it perfect for learning the basics of autograd.

---

## **How It Works**

Minigrad operates using:
- **Forward Pass**: Tensor operations build a computational graph, storing information about inputs, outputs, and the chain of operations.
- **Backward Pass**: Gradients are propagated in reverse through the graph to compute derivatives.

This two-step process mimics what happens during training in machine learning models.

---

## **Installation**

Ensure you have **Python 3.x** and **NumPy** installed:

```bash
pip install numpy
```

Clone the repository to get started:

```bash
git clone https://github.com/yourusername/minigrad.git
cd minigrad
```

Run the code:

```bash
python Minigrad.py
```

---

## **Core Components**

### 1. **Tensor Class**
The `Tensor` class represents data (scalars, vectors, or matrices) and manages automatic differentiation.

#### **Initialization**
```python
def __init__(self, data, requires_grad=False)
```
- **`data`**: Input data (float, int, list, or NumPy array).
- **`requires_grad`**: Indicates if the tensor should track gradients.

Example:
```python
x = Tensor(2.0, requires_grad=True)
```

#### **Attributes**
- **`data`**: Holds the tensor’s values (NumPy array).
- **`grad`**: Stores gradients (initialized as `None`).
- **`requires_grad`**: Tracks if gradients should be calculated.

---

### 2. **Mathematical Operations**
Minigrad supports the following operations:
- **Addition** (`+`): `x + y`
- **Subtraction** (`-`): `x - y`
- **Multiplication** (`*`): `x * y`
- **Division** (`/`): `x / y`
- **Summation** (`sum()`): Sum of all elements.
- **Mean** (`mean()`): Average of all elements.

Each operation:
1. Creates a new tensor.
2. Links inputs and outputs in the computational graph.
3. Prepares for backpropagation.

---

### 3. **Backward Pass**
The `backward()` method computes gradients for tensors involved in a computation.

#### `backward(grad=None)`
- **`grad`**: Initial gradient passed from the final output (default is 1 for scalar outputs).

Example:
```python
z = x * y  # Forward pass
z.backward()  # Backward pass
print(x.grad, y.grad)  # Gradients of x and y
```

---

### 4. **Gradient Reset**
Gradients accumulate over multiple operations unless explicitly reset using `zero_grad()`.

Example:
```python
x.zero_grad()
y.zero_grad()
```

---

## **Examples**

### **1. Basic Operations**
Perform basic tensor operations and compute gradients.

```python
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

# Perform operations
z = x * y  # Forward pass
z.backward()  # Backward pass

# Results
print(f"Result: {z.data}")  # 6.0
print(f"Gradient of x: {x.grad}")  # 3.0
print(f"Gradient of y: {y.grad}")  # 2.0
```

---

### **2. Linear Regression**
Demonstrates training a linear regression model using gradient descent.

```python
# Initialize parameters
w = Tensor(0.0, requires_grad=True)
b = Tensor(0.0, requires_grad=True)

# Data
x_data = Tensor([1.0, 2.0, 3.0, 4.0])
y_data = Tensor([2.0, 4.0, 6.0, 8.0])

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    y_pred = w * x_data + b
    loss = ((y_pred - y_data) ** 2).sum() / len(x_data)

    # Reset gradients
    w.zero_grad()
    b.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient descent
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

---

## **Advantages of Minigrad**

1. **Clarity**: Simplified implementation without unnecessary abstraction.
2. **Learning Tool**: Ideal for beginners to understand backpropagation and autograd.
3. **Lightweight**: Uses only essential features, making it easy to modify or extend.

---

## **Limitations**

1. **Performance**: Not optimized for large-scale computations.
2. **Functionality**: Limited to basic tensor operations.
3. **Error Handling**: Minimal validation for input data types and shapes.

---

## **Contributing**

Feel free to contribute to **Minigrad**! Whether it’s fixing bugs, adding features, or improving documentation, contributions are welcome.

### Steps to contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Make improvements and test them.
4. Submit a pull request.

---

## **License**

This project is licensed under the **MIT License**, allowing you to freely use, modify, and distribute the code.

---
