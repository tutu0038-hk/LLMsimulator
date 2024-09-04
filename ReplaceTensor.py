import torch
from tempfile import TemporaryDirectory
from typing import Tuple, Optional

from torch import nn, Tensor
import torch.distributed
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

global NET_INITTED
NET_INITTED = True
rank = 0
debugging = False

from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    _DistributedBackendOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    DebugLevel,
    get_debug_level,
    Work,
    _register_process_group,
    _resolve_process_group,
    _unregister_all_process_groups,
    _unregister_process_group,
)

from LLMsimulator_pb2 import CommunicatorInput
from LLMsimulator_grpc import GreeterStub
import asyncio
from grpclib.client import Channel
import time
import torch.distributed.distributed_c10d as c10d
pool = [0.0]

class FakeTensor(Tensor):
    def test():
        return None
    
    def __init__(self, sizes):
        self.shape = sizes
            
    def __add__(self, other):
        return self
    
    def __sub__(self, other):
        return self
    

TB = 1024 * 1024 * 1024
GB = 1024 * 1024
TFlops = 1e12
memorySpeed = 1 * TB
communicationSpeed = 64 * GB
computationSpeed = 83 * TFlops
memory = 24 * GB

from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    FakeTensor,
)

memoryPool = {}
def FetchFakeTensor(outDim):
    code = str(outDim)
    if code in memoryPool:
        return memoryPool[code]
    else:
        memoryPool[code] = torch.ones(outDim, requires_grad = False)  
        return memoryPool[code]

async def commnicatorasync(r, t, tp) -> None:
        async with Channel('127.0.0.1', 50051) as channel:
            greeter = GreeterStub(channel)
            reply = await greeter.communicator(CommunicatorInput(rank = r, time = pool[0], type = tp))
       
def _Linear(input: Tensor, weight, bias: Optional[Tensor] = None, 
    scale: Optional[float] = None, zero_point: Optional[int] = None):
    dim = input.dim()
    outDim = [0] * dim
    totalDim = 1
    for i in range(dim):
        outDim[i] = input.shape[i]
        totalDim *= outDim[i]
    flops = totalDim * weight.shape[0]
    totalDim = totalDim / outDim[dim-1] * weight.shape[0]
    outDim[dim - 1] = weight.shape[0]
    flops *= 2 # constant
    if bias != None:
        flops += totalDim #Ax + b
    pool[0] += flops / computationSpeed
    output = FetchFakeTensor(outDim)
    return output

def _matmul(tensorA, tensorB):
    #time0 = time.time()
    shapeA = tensorA.size()
    shapeB = tensorB.size()
    dim1 = len(shapeA)
    dim2 = len(shapeB)
    outDim = [0] * dim1
    flops = 1
    for i in range(dim1):
        outDim[i] = shapeA[i]
        flops *= shapeA[i]
    flops *= shapeB[dim2 - 1]
    flops *= 2 #constant
    outDim[dim1 - 1] = shapeB[dim2 - 1]
    output = FetchFakeTensor(outDim)
    pool[0] += flops / computationSpeed
    #time1 = time.time()
    #print("_matmul : ", time1 - time0)
    return output

def _embeddings(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    shape = input.size()
    embedding_dim = weight.shape[len(weight.size()) - 1]
    dim = len(shape)
    outDim = [0] * dim
    for i in range(dim):
        outDim[i] = shape[i]
    outDim.append(embedding_dim)
    output = FetchFakeTensor(outDim)
    return output

def _bmm(A, B):
    shapeA = A.size()
    shapeB = B.size()
    flops = shapeA[1] * shapeA[2] * shapeB[2] * 2
    pool[0] += flops / computationSpeed
    output = torch.ones(shapeA[0],shapeA[1],shapeB[2])
    return output

def _tocuda(
        self,
        memory_format=torch.preserve_format,
        process_group=None):
    output = FetchFakeTensor(self.shape)
    return output


def _world_size(group: Optional[ProcessGroup] = None):
    return world_size

def _get_rank(group: Optional[ProcessGroup] = None):
    return rank

def all_reduce_md(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        sizes = tensor.size()
        flops = 1
        for i in sizes:
            flops *= i
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size =  c10d._get_group_size(pg)
    flops *= 2 * (size - 1) / size
    asyncio.run(commnicatorasync(rank, flops, 1))
    pool[0] = 0
    return None

def all_gather_md(tensor_list, tensor, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        sizes = tensor.size()
        flops = 1
        for i in sizes:
            flops *= i
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    asyncio.run(commnicatorasync(rank, flops, 2))
    tensor_list = [tensor for _ in range(size)]
    pool[0] = 0
    return None

def barrier_md(group, async_op=False, device_ids=None):
    """
    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().
    """
    return None

def _send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    return None

def _recv(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    return None

def _all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    return None

def _broadcast(tensor, src, group=None, async_op=False):
    return None

def _reduce_scatter(raw):
    return None

def _todevice(self, *args, **kwargs):
    #print("to device")
    # if self._local_shards:
    #     current_device = self._local_shards[0].tensor.device
    # elif self._process_group._get_backend_name() == "gloo":
    #     current_device = torch.device("cpu")
    # else:
    #     current_device = torch.device(torch.cuda.current_device())
    current_device = self.device
    current_dtype = self.dtype
    device_to = current_device
    dtype_to = current_dtype
    if len(args) == 1:
        if isinstance(args[0], torch.dtype):
            dtype_to = args[0]
        elif isinstance(args[0], torch.device):
            device_to = args[0]
        elif isinstance(args[0], (str, int)):
            device_to = torch.device(args[0])
        elif isinstance(args[0], torch.Tensor):
            dtype_to = args[0].dtype
            device_to = args[0].device
        else:
            raise RuntimeError(f"ShardedTensor.to() have wrong arguments: {args}")
    elif len(args) == 2:
        device_to, dtype_to = args
    else:
        dtype_to = kwargs.get("dtype", current_dtype)
        device_to = kwargs.get("device", current_device)

    device_to = torch.device(device_to) if isinstance(device_to, (str, int)) else device_to

    if device_to.type == "cuda":
        # if device_to set to cuda, set to current device even
        # if user specify the device index.
        current_idx = torch.cuda.current_device()
        if device_to.index != current_idx:
            warnings.warn("ShardedTensor.to only move tensor to its current device"
                            "If you want to put to different device, use `reshard` instead.")
        device_to = torch.device(current_idx)
        flops = 1
        for i in self.shape:
            flops *= i
        pool[0] += flops * 4.0 / computationSpeed / 1024
        print("to decive :", flops)
        return 

    copy_tensor = kwargs.get("copy", False)
    non_blocking = kwargs.get("non_blocking", False)
    memory_format = kwargs.get("memory_format", torch.preserve_format)
    process_group = kwargs.get("process_group", None)

    return self

def clearpool():
    if pool[0] > 0:
        asyncio.run(commnicatorasync(rank, pool[0], 0))
        pool[0] = 0

def init(rank0, world_size0):
    global rank
    global world_size
    rank = rank0
    world_size = world_size0
    torch.bmm = _bmm
    nn.functional.linear = _Linear
    torch.matmul = _matmul
    nn.functional.embedding = _embeddings

    #distributed group
    torch.distributed.get_world_size =  _world_size
    torch.distributed.get_rank = _get_rank
    torch.distributed.get_group_rank = _get_rank

    #cost data transforming time
    torch.Tensor.to = _todevice
    #torch.Tensor.cpu = _tocuda

    #torch.Tensor.cuda = Rp(torch.Tensor.cuda, _cuda)
    #commnication part
    torch.distributed.all_gather = all_gather_md
    torch.distributed.all_reduce = all_reduce_md
    torch.distributed.broadcast = _broadcast
    torch.distributed.barrier = barrier_md
    torch.distributed.send = _send
    torch.distributed.recv = _recv
    torch.distributed.all_to_all = _all_to_all
    torch.distributed.reduce_scatter = _reduce_scatter

