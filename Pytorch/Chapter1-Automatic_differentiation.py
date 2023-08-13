import torch
import numpy as np

def calc_gradient(x):
    y = 2 * torch.dot(x,x)
    print("y:",y)
    y.backward()
    print("x.grad:",x.grad)

if __name__ == "__main__":
    print("Gradient calculation")
    x = torch.tensor([0,1,2,3], requires_grad=True, dtype=torch.float32)
    calc_gradient(x)