import numpy as np
from typing import Optional, Union, List, Callable, Tuple

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float, int], requires_grad: bool = False):
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._grad_fn: Optional[Callable] = None
        self._inputs: List[Tensor] = []
        self._is_leaf = True
        
        if requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Calling backward on a tensor that doesn't require gradients")

        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        visited = set()
        topo_order = []

        def build_topo(node: 'Tensor') -> None:
            if node not in visited and node.requires_grad:
                visited.add(node)
                for input_node in node._inputs:
                    if input_node.requires_grad:
                        build_topo(input_node)
                topo_order.append(node)

        build_topo(self)
        
        for node in reversed(topo_order):
            if node._grad_fn is not None:
                input_grads = node._grad_fn(node.grad)
                for input_tensor, input_grad in zip(node._inputs, input_grads):
                    if input_tensor.requires_grad:
                        sum_axes = tuple(range(len(input_grad.shape) - len(input_tensor.data.shape)))
                        if sum_axes:
                            input_grad = np.sum(input_grad, axis=sum_axes)
                        
                        for i, (grad_dim, tensor_dim) in enumerate(zip(input_grad.shape, input_tensor.data.shape)):
                            if grad_dim != tensor_dim:
                                input_grad = np.sum(input_grad, axis=i, keepdims=True)
                        
                        if input_tensor.grad is None:
                            input_tensor.grad = input_grad
                        else:
                            input_tensor.grad += input_grad

    def _create_binary_operation(self, other: Union['Tensor', float, int], 
                               op: Callable, 
                               gradient_fn: Callable) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(op(self.data, other.data), requires_grad=requires_grad)

        if requires_grad:
            result._is_leaf = False
            result._inputs = [self, other]
            result._grad_fn = lambda grad: gradient_fn(grad, self.data, other.data)

        return result

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._create_binary_operation(
            other,
            lambda x, y: x + y,
            lambda grad, x, y: (grad, np.broadcast_to(grad, x.shape))
        )

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._create_binary_operation(
            other,
            lambda x, y: x * y,
            lambda grad, x, y: (grad * y, grad * x)
        )

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._create_binary_operation(
            other,
            lambda x, y: x - y,
            lambda grad, x, y: (grad, -grad)
        )

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._create_binary_operation(
            other,
            lambda x, y: x / y,
            lambda grad, x, y: (grad / y, -grad * x / (y * y))
        )

    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        result = Tensor(self.data ** exponent, requires_grad=self.requires_grad)
        if self.requires_grad:
            result._is_leaf = False
            result._inputs = [self]
            result._grad_fn = lambda grad: (grad * exponent * self.data ** (exponent - 1),)
        return result

    def sum(self) -> 'Tensor':
        result = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            result._is_leaf = False
            result._inputs = [self]
            result._grad_fn = lambda grad: (np.ones_like(self.data) * grad,)
            
        return result

    def mean(self) -> 'Tensor':
        result = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            result._is_leaf = False
            result._inputs = [self]
            result._grad_fn = lambda grad: (np.ones_like(self.data) * grad / self.data.size,)
            
        return result

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad})"

# Test functions
def test_linear_regression():
    print("\nStarting Linear Regression Test:")
    
    # Initialize parameters
    w = Tensor(0.0, requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    
    # Training data
    x_data = Tensor([1.0, 2.0, 3.0, 4.0])
    y_data = Tensor([2.0, 4.0, 6.0, 8.0])
    
    # Training loop
    learning_rate = 0.01
    print(f"Initial parameters - w: {w.data}, b: {b.data}")
    
    for epoch in range(1000):
        # Forward pass
        y_pred = w * x_data + b
        loss = ((y_pred - y_data) ** 2).mean()  # MSE loss
        
        # Reset gradients
        w.zero_grad()
        b.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data.item():.6f}, w = {w.data[0]:.6f}, b = {b.data[0]:.6f}")
    
    print(f"\nFinal parameters - w: {w.data}, b: {b.data}")
    print("Expected values - w: 2.0, b: 0.0")

def test_basic_operations():
    print("\nTesting Basic Operations:")
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y
    z.backward()
    print(f"x * y = {z.data}, dx = {x.grad}, dy = {y.grad}")
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)
    z = (x * y).sum()
    z.backward()
    print(f"sum(x * y) = {z.data}, dx = {x.grad}, dy = {y.grad}")

if __name__ == "__main__":
    test_basic_operations()
    test_linear_regression()