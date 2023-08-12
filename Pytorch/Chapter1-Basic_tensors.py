#All about tesors
import torch 

class BasicsOperation:
    def __init__(self) -> None:
        pass
    '''A tensor represents a (possibly multi-dimensional) array of numerical values. With one axis, a tensor is called a vector. 
    With two axes, a tensor is called a matrix. With x>2 axes, we drop the specialized names and just refer to the object as a xth order tensor. '''
    def create_tensor(size_tensor):
        x = torch.arange(size_tensor, dtype=torch.float32)
        print("Tensor x: ",x)
        return x
    def shape_tensor(tensor):
        print(tensor.shape_tensor())
    def dtype_tensor(tensor):
        print(tensor.dtype_tensor())    
    def device_tensor(tensor):
        print(tensor.device_tensor())
    def sum_tensor(tensor):
        print(tensor.sum_tensor())

    def reshape_tensor(tensor_name,size1,size2):
        try:
            tensor_name.reshape(size1,size2)
        except Exception as e:
            print("Error: ",e)
    def zero_tensor(size1,size2):
        tensor = torch.zeros(size1,size2)
        print("Zero tensor: ",tensor)
        return tensor
    def one_tensor(size1,size2):
        tensor = torch.ones(size1,size2)
        print("One tensor: ",tensor)
        return tensor
    def random_tensor(size1,size2):
        tensor = torch.rand(size1,size2)
        print("Random tensor: ",tensor)
        return tensor
    def random_tensor_with_range(size1,size2,start,end):
        tensor = torch.randint(size1,size2,start,end)
        print("Random tensor with range: ",tensor)
        return tensor
    def random_tensor_with_range_and_dtype(size1,size2,start,end,dtype):
        tensor = torch.randint(size1,size2,start,end,dtype)
        print("Random tensor with range and dtype: ",tensor)
        return tensor
    def slice_tensor(tensor,start,end):
        tensor = tensor[start:end]
        print("Slice tensor: ",tensor)
        return tensor
    def slice_tensor_with_step(tensor,start,end,step):
        tensor = tensor[start:end:step]
        print("Slice tensor with step: ",tensor)
        return tensor
    def slice_tensor_with_negative_indexing(tensor,start,end,step):
        tensor = tensor[start:end:step]
        print("Slice tensor with negative indexing: ",tensor)
        return tensor
    def slice_tensor_with_negative_indexing_and_reverse(tensor,start,end,step):
        tensor = tensor[start:end:step]
        print("Slice tensor with negative indexing and reverse: ",tensor)
        return tensor  
    def expand_tensor(tensor,size1,size2):
        tensor = tensor.expand(size1,size2)
        print("Expand tensor: ",tensor)
        return tensor    
    def operate_tensor(tensor1,tensor2,operation):
        if operation == "add":
            tensor = tensor1 + tensor2
        elif operation == "sub":
            tensor = tensor1 - tensor2
        elif operation == "mul":
            tensor = tensor1 * tensor2
        elif operation == "div":
            tensor = tensor1 / tensor2
        elif operation == "pow":
            tensor = tensor1 ** tensor2
        elif operation == "matmul":
            tensor = tensor1 @ tensor2
        print("Operate tensor: ",tensor)
        return tensor
    def cat_tensor(tensor1,tensor2,dim):
        if dim == 0:
            tensor = torch.cat((tensor1,tensor2),dim=0) 
        elif dim == 1:
            tensor = torch.cat((tensor1,tensor2),dim=1)
        print("Cat tensor: ",tensor)
        return tensor
    def write_tensor(tensor,position,value):
        tensor[position] = value
        print("Write tensor: ",tensor)
        return tensor
    def create_tensor_from_list(list):  
        tensor = torch.tensor(list)
        print("Tensor from list: ",tensor)
        return tensor
    def create_tensor_from_numpy(numpy_array):
        tensor = torch.from_numpy(numpy_array)
        print("Tensor from numpy: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor(tensor):
        tensor = torch.tensor(tensor)
        print("Tensor from another tensor: ",tensor)
        return tensor       
    def create_tensor_from_another_tensor_with_new_dtype(tensor,dtype):
        tensor = torch.tensor(tensor,dtype=dtype)
        print("Tensor from another tensor with new dtype: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device(tensor,dtype,device):
        tensor = torch.tensor(tensor,dtype=dtype,device=device)
        print("Tensor from another tensor with new dtype and device: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad(tensor,dtype,device,requires_grad):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad)
        print("Tensor from another tensor with new dtype and device and requires_grad: ",tensor)
        return tensor       
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory(tensor,dtype,device,requires_grad,pin_memory):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking(tensor,dtype,device,requires_grad,pin_memory,non_blocking):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy_and_memory_format(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy,memory_format):         
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy,memory_format=memory_format)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy and memory_format: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy_and_memory_format_and_grad(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy,memory_format,grad):         
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy,memory_format=memory_format,grad=grad)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy and memory_format and grad: ",tensor)
        return tensor   
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy_and_memory_format_and_grad_fn(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy,memory_format,grad,grad_fn):         
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy,memory_format=memory_format,grad=grad,grad_fn=grad_fn)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy and memory_format and grad and grad_fn: ",tensor)
        return tensor
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy_and_memory_format_and_grad_fn_and_is_leaf(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy,memory_format,grad,grad_fn,is_leaf):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy,memory_format=memory_format,grad=grad,grad_fn=grad_fn,is_leaf=is_leaf)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy and memory_format and grad and grad_fn and is_leaf: ",tensor)
        return tensor
    def create_tensor_from_another_tensor_with_new_dtype_and_device_and_requires_grad_and_pin_memory_and_non_blocking_and_copy_and_memory_format_and_grad_fn_and_is_leaf_and_kwargs(tensor,dtype,device,requires_grad,pin_memory,non_blocking,copy,memory_format,grad,grad_fn,is_leaf,**kwargs):
        tensor = torch.tensor(tensor,dtype=dtype,device=device,requires_grad=requires_grad,pin_memory=pin_memory,non_blocking=non_blocking,copy=copy,memory_format=memory_format,grad=grad,grad_fn=grad_fn,is_leaf=is_leaf,**kwargs)
        print("Tensor from another tensor with new dtype and device and requires_grad and pin_memory and non_blocking and copy and memory_format and grad and grad_fn and is_leaf and kwargs: ",tensor)
        return tensor
    

    
    