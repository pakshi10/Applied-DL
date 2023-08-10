import torch

class LinearAlgebra():
    def __init__(self) -> None:
        pass
    def scalar_operation(self,tensor1,tensor2):
        print("Scalar operation")
        print("tensor1 + tensor2 = ", tensor1 + tensor2)    
        print("tensor1 * tensor2 = ", tensor1 * tensor2)
        print("tensor1 / tensor2 = ", tensor1 / tensor2)
        print("tensor1 ** tensor2 = ", tensor1 ** tensor2)
    def tensor_operation(self,tensor1,tensor2):
        print("Tensor operation")
        print("tensor1 + tensor2 = ", tensor1 + tensor2)
        print("tensor1 * tensor2 = ", tensor1 * tensor2)    
        print("tensor1 / tensor2 = ", tensor1 / tensor2)
        print("tensor1 ** tensor2 = ", tensor1 ** tensor2)
        print("tensor1 @ tensor2 = ", tensor1 @ tensor2)
        print("tensor1 sum = ", tensor1.sum())
        print("tensor1 shape = ", tensor1.shape)
        print("tensor1 mean = ", tensor1.float().mean())
        print("tensor1 std = ", tensor1.float().std())
        print("tensor1 max = ", tensor1.max())
        print("tensor1 min = ", tensor1.min())
        print("tensor1.argmax = ", tensor1.argmax())
        print("tensor1.argmin = ", tensor1.argmin())
        print("tensor1 transpose = ", tensor1.transpose(0, 1))
        print("tensor1.T = ", tensor1.T)


    def matrix_operation(self,tensor1,tensor2):
        print("Matrix operation")
        print("tensor1 = ", tensor1)
        print("tensor2 = ", tensor2)
        print("tensor1.shape = ", tensor1.shape)
        print("tensor2.shape = ", tensor2.shape)
        print("tensor1 @ tensor2 = ", tensor1 @ tensor2)
        print("tensor2 @ tensor1 = ", tensor2 @ tensor1)
        print("tensor1.mm(tensor2) = ", tensor1.mm(tensor2))
        print("tensor2.mm(tensor1) = ", tensor2.mm(tensor1))
        print("tensor1.matmul(tensor2) = ", tensor1.matmul(tensor2))
        print("tensor2.matmul(tensor1) = ", tensor2.matmul(tensor1))
        print("tensor1.T = ", tensor1.T)
        print("tensor2.T = ", tensor2.T)
        print("tensor1.transpose(0, 1) = ", tensor1.transpose(0, 1))
        print("tensor2.transpose(0, 1) = ", tensor2.transpose(0, 1))
        print("tensor1 @ tensor1.T = ", tensor1 @ tensor1.T)
        print("tensor2 @ tensor2.T = ", tensor2 @ tensor2.T)
        print("tensor1.mm(tensor1.T) = ", tensor1.mm(tensor1.T))
        print("tensor2.mm(tensor2.T) = ", tensor2.mm(tensor2.T))
        print("tensor1.matmul(tensor1.T) = ", tensor1.matmul(tensor1.T))
        print("tensor2.matmul(tensor2.T) = ", tensor2.matmul(tensor2.T))
        print("torch.matmul(tensor1, tensor1.T) = ", torch.matmul(tensor1, tensor1.T))
        print("torch.matmul(tensor2, tensor2.T) = ", torch.matmul(tensor2, tensor2.T))
        print("tensor1 @ tensor1 = ", tensor1 @ tensor1)
        print("tensor2 @ tensor2 = ", tensor2 @ tensor2)
        print("tensor1.mm(tensor1) = ", tensor1.mm(tensor1))
        print("tensor2.mm(tensor2) = ", tensor2.mm(tensor2))
        print("tensor1.matmul(tensor1) = ", tensor1.matmul(tensor1))
        #norm l1 and l2
        print("tensor1.norm() = ", tensor1.float().norm())
        print("tensor2.norm() = ", tensor2.float().norm())
        print("tensor1.norm(p=1) = ", tensor1.float().norm(p=1))
        print("tensor2.norm(p=1) = ", tensor2.float().norm(p=1))
        print("tensor1.norm(p=2) = ", tensor1.float().norm(p=2))
        print("tensor2.norm(p=2) = ", tensor2.float().norm(p=2))

if __name__ == "__main__":
    tensor1 = torch.tensor([[1, 2],[3, 4]])
    tensor2 = torch.tensor([[5, 6],[7, 8]])
    tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor4 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("tensor1 = ", tensor1)
    print("tensor2 = ", tensor2)
    linear_algebra = LinearAlgebra()
    linear_algebra.scalar_operation(tensor1,tensor2)
    linear_algebra.tensor_operation(tensor1,tensor2)
    linear_algebra.matrix_operation(tensor3,tensor4)




