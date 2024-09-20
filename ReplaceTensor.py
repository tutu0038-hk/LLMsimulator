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
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        import logging as not_implemented_log
    else:
        raise e
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
from torch.utils._stats import count
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, TypeVar
from torch.multiprocessing.reductions import StorageWeakRef
from weakref import ReferenceType
from torch._prims_common import ShapeType
pool = [0.0]

Trace = []

def WriteRecord(type, flops):
    Trace.append([type, flops])

TB = 1024 * 1024 * 1024
GB = 1024 * 1024
TFlops = 1e12
memorySpeed = 1 * TB
communicationSpeed = 64 * GB
computationSpeed = 83 * TFlops
memory = 24 * GB

backupReshape = torch.Tensor.reshape
backupView = torch.Tensor.view

from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)

from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)

from torch._guards import Source

class FakeTensorWithNoData(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """
    Fakeshape: ShapeType
    fake_device: torch.device
    fake_mode: "FakeTensorMode"
#   constant: Optional[torch.Tensor]

    # This memorizes the unbacked SymInt representing the number of nonzero
    # elements in this tensor.  This is helpful if you do something like
    # x[mask] and y[mask]; mask.nonzero() gets repeatedly called and should
    # give a consistent unbacked SymInt.  It needs to be invalidated in the
    # same way constant is.
    # TODO: Generalize this as needed, e.g., into a trie of memos
    _nonzero_memo: Optional[torch.SymInt]
    _nonzero_memo_vc: Optional[int]

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE
    
    @property
    def nonzero_memo(self):
        if self._nonzero_memo is None:
            return None
        # Version counter based tracking isn't 100% sound but it's close
        # enough
        if self._nonzero_memo_vc != self._version:
            self._nonzero_memo = None
            return None
        return self._nonzero_memo

    @property
    def device(self):
        return torch.device("cpu")

    def __getitem__(self, key):
        #print("Fakeshape", self.Fakeshape)
        #print("args= ", key)
        if isinstance(key, slice):
            input = (key, )
        else:
            input = key

        length = len(self.Fakeshape)
        outDim = [0] * length
        i = 0
        cnt = 0
        for slices in input:
            if not slices is None:
                if slices == -1:
                    outDim.pop(i)
                    length -= 1
                    cnt += 1
                else:
                    #print(slices)
                    outDim[i] = len(range(*slices.indices(self.Fakeshape[i])))
                    i += 1
        
        for j in range(i, length):
            outDim[j] = self.Fakeshape[j + cnt]

        out = FakeTensorWithNoData(outDim)
        return out
    
    def __mul__(self, scalar):
        return torch.matmul(self, scalar)
    
    def __setitem__(self, key, value):
        flops = 1
        for i in value.Fakeshape:
            flops *= i
        pool[0] += flops * 4.0 / memorySpeed / 1024

    def __truediv__(self, other):
        return self
    
    def __add__(self, other):
        return self
    
    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch__dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below
    # NOTE: this probably will not do the right thing for backends
    # that have dispatch keys which are higher than the "meta" key:
    # https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h#L189

    # We don't support named tensors; graph break
    @property
    def names(self):
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    def float(self):
        return self
    
    def dim(self):
        return len(self.Fakeshape)
    
    def Fakeshape(self):
        return self.Fakeshape
        
    @staticmethod
    def __new__(cls, dim):
        self = super().__new__(cls)
        self.Fakeshape = list(dim)
        return self

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not torch.Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break

        assert fake_mode is not None

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FAKE
        )
        if maybe_cur_fake_mode:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_mode,
                maybe_cur_fake_mode,
            )
            return NotImplemented

        with fake_mode:  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(func, flat_args) -> Tuple[torch.device, bool]:
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        def cpu_zero_dim(t):
            return t.device.type == "cpu" and t.dim() == 0

        def merge_devices(t):
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise RuntimeError(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        for arg in flat_args:
            merge_devices(arg)

        # some functions that allow Python numbers to bind to Tensors
        # if we have failed to find a device, and we're running one of these operators,
        # we must have scalar only inputs
        if should_allow_numbers_as_tensors(func) and common_device is None:
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        assert common_device is not None, f"Could not find common device for {func}"

        return common_device, has_scalar_only_inputs

    # We must handle tolist in a special way for FakeTensors here in the case
    # where tolist is called from torch dispatch for tensor subclasses.
    # Ordinarily, if a program calls .tolist compiling still works because there is
    # special handling in dynamo, but for tensor subclasses if .tolist is called
    # inside torch dispatch, the .tolist call may be directly on a FakeTensor.
    # This would result in an error since wrapper subclasses don't have storage.
    # To avoid this, we handle the FakeTensor case by (1) specializing on the size
    # of the tensor to create the output Python list, and (2) creating unbacked
    # symints for each element of the list.
    def tolist(self):
        assert self.dim() == 1, "NYI for higher dims"
        shape_env = self.fake_mode.shape_env
        out = []
        # Specialize on the length of the list
        for _ in range(self.Fakeshape[0]):
            s = shape_env.create_unbacked_symint()
            # max value?
            torch._constrain_as_size(s, min=2)
            out.append(s)
        return out


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverterWithNoData:
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: Dict[StorageWeakRef, List[ReferenceType]]

    def __init__(self):
        self.meta_converter = MetaConverter()

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor):
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor):
        assert not isinstance(tensor, FakeTensor)

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def _get_memo(self, t):
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t, v):
        tid = self.meta_converter.describer.get_tensor_id(t)
        self.meta_converter.tensor_memo[tid] = v

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def from_real_tensor(
        self,
        fake_mode,
        basicDim,
        shape = None,
        make_constant=False,
        shape_env=None,
        *,
        source=None,
        symbolic_context=None,
    ):
        # # see note [Tensor Fakification and Symbol Caching]
        # if not symbolic_context and not source and shape_env:
        #     if tracing_context := torch._guards.TracingContext.try_get():
        #         if t in tracing_context.tensor_to_context:
        #             symbolic_context = tracing_context.tensor_to_context[t]
        #             source = symbolic_context.tensor_source

        # # maybe_memo = self._get_memo(t)
        # # if maybe_memo is not None:
        # #     return maybe_memo
        # # existing_device = t.device
        # # # not yet supported in metatensors
        # if t.is_quantized:
        #     raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        # if type(t) is torch.nn.Parameter:
        #     assert not make_constant
        out = FakeTensorWithNoData(
                basicDim,
            )
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(self, fake_mode, t, device):
        assert (
            t.device.type == "meta"
        ), f"tensor's device must be `meta`, got {t.device.type} instead"
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensorWithNoData(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out
    
def _from_tensor(
        self,
        basicDim,
        *,
        static_shapes=None,
        source: Optional[Source] = None,
        symbolic_context=None,
    ):
        shape_env = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert (
                symbolic_context is None
            ), "cannot set both static_shapes and symbolic_context"
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            basicDim,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
        )

FakeTensorMode.from_tensor = _from_tensor
mode = FakeTensorMode(allow_non_fake_inputs = True)
mode.fake_tensor_converter = FakeTensorConverterWithNoData()

memoryPool = {}
def FetchFakeTensor(outDim):
    return mode.from_tensor(outDim)

async def commnicatorasync(r, t, tp) -> None:
    async with Channel('127.0.0.1', 50051) as channel:
        greeter = GreeterStub(channel)
        reply = await greeter.communicator(CommunicatorInput(rank = r, time = pool[0], type = tp))

def MakeFake(self):
    if (self.__class__.__name__ == "FakeTensor"):
        return self
    else:
        return mode.from_tensor(self.shape)

def _flatten(self, start_dim = 0, end_dim = -1):
    shape = self.Fakeshape
    if end_dim == -1:
        end_dim = len(shape)
    new_len = len(shape) - (end_dim - start_dim) + 1
    outDim = [0] * new_len
    total = 1
    for i in range(start_dim, end_dim):
        total *= shape[i]
    for i in range(start_dim):
        outDim[i] = shape[i]
    outDim[start_dim] = total
    self.Fakeshape = outDim
    #print(shape, "start_dim = ", start_dim, "end_dim = ", end_dim)
    #print("outDim = ", outDim)
    return self

def _Linear(input: Tensor, weight, bias: Optional[Tensor] = None, 
    scale: Optional[float] = None, zero_point: Optional[int] = None):
    weight = MakeFake(weight)
    dim = len(input.Fakeshape)
    outDim = [0] * dim
    totalDim = 1
    for i in range(dim):
        outDim[i] = input.Fakeshape[i]
        totalDim *= outDim[i]
    flops = totalDim * weight.Fakeshape[0]
    totalDim = totalDim / outDim[dim-1] * weight.Fakeshape[0]
    outDim[dim - 1] = weight.Fakeshape[0]
    flops *= 2 # constant
    if bias != None:
        flops += totalDim #Ax + b
    pool[0] += flops / computationSpeed
    output = FetchFakeTensor(outDim)
    return output

def _matmul(tensorA, tensorB):
    shapeA = tensorA.Fakeshape
    shapeB = tensorB.Fakeshape
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
    return output

def _embeddings(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    weight = MakeFake(weight)
    Fakeshape = input.Fakeshape
    embedding_dim = weight.Fakeshape[len(weight.Fakeshape) - 1]
    dim = len(Fakeshape)
    outDim = [0] * dim
    for i in range(dim):
        outDim[i] = Fakeshape[i]
    outDim.append(embedding_dim)
    output = FetchFakeTensor(outDim)
    return output

def _view_as_real(self):
    #print("_view_as_real", self.Fakeshape)
    self.Fakeshape.pop()
    #print("_view_as_real", self.Fakeshape)
    return self

def _view_as_complex(self):
    #print("_view_as_complex", self.Fakeshape)
    self.Fakeshape.append(2)
    #print("_view_as_complex", self.Fakeshape)
    return self 

def _view_as(self, other):
    self.Fakeshape = other.Fakeshape
    return self

def _reshape(self, *Fakeshape: ShapeType):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        #print("reshape", self.Fakeshape)
        default = True
        for shapes in Fakeshape:
            if shapes == -1:
                default = False
        if default:
            dim = len(Fakeshape)
            self.Fakeshape = [0] * dim
            for i in range(dim):
                self.Fakeshape[i] = Fakeshape[i]
        else:
            totalDim = 1
            for shapes in self.Fakeshape:
                totalDim *= shapes
            dim = len(Fakeshape)
            self.Fakeshape = [0] * dim
            for i in range(dim):
                if Fakeshape[i] != -1:
                    totalDim /= Fakeshape[i]
                    self.Fakeshape[i] = Fakeshape[i]
            for i in range(dim):
                if Fakeshape[i] == -1:
                    self.Fakeshape[i] = totalDim
        #print("reshape", ShapeType)
        #print("reshape", self.Fakeshape)
        return self
    else:
        return backupReshape(self, Fakeshape)

def _view(self, *Fakeshape: ShapeType):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        #print(self.Fakeshape)
        default = True
        for shapes in Fakeshape:
            if shapes == -1:
                default = False
        if default:
            dim = len(Fakeshape)
            self.Fakeshape = [0] * dim
            for i in range(dim):
                self.Fakeshape[i] = Fakeshape[i]
        else:
            totalDim = 1
            for shapes in Fakeshape:
                totalDim *= shapes
            dim = len(Fakeshape)
            self.Fakeshape = [0] * dim
            for i in range(dim):
                if Fakeshape[i] != -1:
                    totalDim /= Fakeshape[i]
                    self.Fakeshape[i] = Fakeshape[i]
            for i in range(dim):
                if Fakeshape[i] == -1:
                    self.Fakeshape[i] = totalDim
        #print(ShapeType)
        #print(self.Fakeshape)
        return self
    else:
        return backupView(self, Fakeshape)

def _expand(self, *Fakeshape: ShapeType):
    self.Fakeshape = list(Fakeshape)
    return self

def _transpose(self, dim0, dim1):
    self.Fakeshape[dim0],self.Fakeshape[dim1] = self.Fakeshape[dim1],self.Fakeshape[dim0]
    return self

def _silu(self, inplace=False):
    return self

def _sort(self, dim=-1, descending=False, stable=False, *, out=None):
    return self, self

def _cunsum(self, dim, *, dtype=None, out=None):
    return self

def _tocuda(
        self,
        memory_format=torch.preserve_format,
        process_group=None):
    self = MakeFake(self)
    output = FetchFakeTensor(self.Fakeshape)
    flops = 1
    for i in self.Fakeshape:
        flops *= i
    pool[0] += flops * 4.0 / memorySpeed / 1024
    return output

def _world_size(group: Optional[ProcessGroup] = None):
    return world_size

def _get_rank(group: Optional[ProcessGroup] = None):
    return rank

def all_reduce_md(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        sizes = tensor.Fakeshape
        flops = 1
        for i in sizes:
            flops *= i
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size =  c10d._get_group_size(pg)
    flops *= 2 * (size - 1) / size
    #asyncio.run(commnicatorasync(rank, flops, 1))
    WriteRecord(0, pool[0])
    WriteRecord(1, flops)
    pool[0] = 0
    return None

def all_gather_md(tensor_list, tensor, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        sizes = tensor.Fakeshape
        flops = 1
        for i in sizes:
            flops *= i
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size / communicationSpeed
    #asyncio.run(commnicatorasync(rank, flops, 2))
    WriteRecord(0, pool[0])
    WriteRecord(2, flops)
    tensor_list = [tensor for _ in range(size)]
    pool[0] = 0
    return None

def barrier_md(group, async_op=False, device_ids=None):
    """
    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().
    """
    asyncio.run(commnicatorasync(rank, 0, 3))
    return None

def _send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    sizes = tensor.Fakeshape
    flops = 1
    for i in sizes:
        flops *= i
    flops /= communicationSpeed
    asyncio.run(commnicatorasync(rank, flops, 4))
    return None

def _recv(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    sizes = tensor.Fakeshape
    flops = 1
    for i in sizes:
        flops *= i
    flops /= communicationSpeed
    asyncio.run(commnicatorasync(rank, flops, 5))
    return None

def _all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    return None

def _broadcast(tensor, src, group=None, async_op=False):
    return None

def _reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    return None

def _todevice(self, *args, **kwargs):
    ##print("to device")
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

    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        return self
    else:
        # current_idx = torch.cuda.current_device()
        # if device_to.index != current_idx:
        #     warnings.warn("ShardedTensor.to only move tensor to its current device"
        #                     "If you want to put to different device, use `reshard` instead.")
        outDim = self.shape
        self = FetchFakeTensor(outDim)
        flops = 1
        for i in self.Fakeshape:
            flops *= i
        pool[0] += flops * 4.0 / memorySpeed / 1024
        #print("to decive :", self.Fakeshape)
        return self

    # if device_to.type == "cuda":
    #     # if device_to set to cuda, set to current device even
    #     # if user specify the device index.
    #     current_idx = torch.cuda.current_device()
    #     if device_to.index != current_idx:
    #         warnings.warn("ShardedTensor.to only move tensor to its current device"
    #                         "If you want to put to different device, use `reshard` instead.")
    #     device_to = torch.device(current_idx)
    #     flops = 1
    #     for i in self.Fakeshape:
    #         flops *= i
    #     pool[0] += flops * 4.0 / memorySpeed / 1024
    #     #print("to decive :", flops)
    #     return self

    # copy_tensor = kwargs.get("copy", False)
    # non_blocking = kwargs.get("non_blocking", False)
    # memory_format = kwargs.get("memory_format", torch.preserve_format)
    # process_group = kwargs.get("process_group", None)

    # return self

def _softmax(self, dim = None):
    return self

def clearpool():
    if pool[0] > 0:
        WriteRecord(0, pool[0])
        pool[0] = 0

def global_fake_mode():
    torch.tensor = FakeTensorWithNoData

def _type_as(self, type):
    return self

def _argmax(self, dim, keepdim = False):   
    shape = self.Fakeshape
    #print("Test : ", shape)
    length = len(shape)
    if dim == -1:
        dim = length - 1
    outDim = [0] * (length - 1)
    idx = 0
    for i in range(length):
        if i != dim:
            outDim[idx] = shape[i]
            idx += 1
    out = torch.ones(outDim)
    #print(outDim)
    return out

def _empty_like(self):
    return self

def _cat(self, dim=0, out=None):
    length = len(self)
    oudDim = self[0].Fakeshape
    oudDim[dim] *= length
    out = FetchFakeTensor(oudDim)
    #print("cattttt = ", dim, oudDim, length)
    return out

def _contiguous(self):
    return self

def _empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format):
    out = FetchFakeTensor(size)
    return out

def _split(self, split_size_or_sections, dim=0):
    self = MakeFake(self)
    if isinstance(split_size_or_sections, int):
        #print("test 222 ", self.Fakeshape, split_size_or_sections, dim)        
        count = int(self.Fakeshape[dim] / split_size_or_sections)
        self.Fakeshape[dim] = split_size_or_sections
        outTensor = FetchFakeTensor(self.Fakeshape)
        #print("test 333 ", self.Fakeshape, split_size_or_sections, dim) 
        out = [outTensor] * count
        return out
    else:
        print("test 213213123", split_size_or_sections)
        return None

def new_tensor(*size):
    out = FetchFakeTensor(size)
    return out

def _Parameter(self):
    out = MakeFake(self)
    return out

def init(rank0, world_size0):
    global rank
    global world_size
    rank = rank0
    world_size = world_size0
    nn.functional.linear = _Linear
    torch.matmul = _matmul
    nn.functional.embedding = _embeddings
    torch.view_as_complex = _view_as_complex
    torch.view_as_real = _view_as_real
    torch.Tensor.reshape = _reshape
    torch.Tensor.view = _view
    torch.Tensor.view_as = _view_as
    torch.Tensor.type_as = _type_as
    torch.Tensor.flatten = _flatten
    torch.Tensor.expand = _expand
    torch.Tensor.transpose = _transpose
    nn.functional.softmax = _softmax
    nn.functional.silu = _silu
    torch.softmax = _softmax
    torch.sort = _sort
    torch.cumsum = _cunsum
    torch.argmax = _argmax
    torch.empty_like = _empty_like
    torch.cat = _cat
    torch.Tensor.contiguous = _contiguous
    #torch.empty = _empty
    torch.split = _split
    #torch.Tensor = new_tensor
    torch.nn.Parameter = _Parameter

    # outDim = [8, 16, 8192]
    # time0 = time.time()
    # tensorA = torch.ones(outDim, requires_grad = False)
    # tensorA = tensorA.view(8, 8192, 16)
    # timenow = time.time()
    # #print("Test : ", timenow - time0)        
    # time0 = time.time()
    # xq = mode.from_tensor(basic, outDim) 
    # xk = mode.from_tensor(basic, outDim)
    
    # #print(*xq.Fakeshape[:-1])
    # #print(*xk.Fakeshape[:-1])
    # time0 = time.time()
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.Fakeshape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.float().reshape(*xk.Fakeshape[:-1], -1, 2))

    # #print(xq.float().reshape(*xq.Fakeshape[:-1], -1, 2).Fakeshape)
    # #print(xk.float().reshape(*xq.Fakeshape[:-1], -1, 2).Fakeshape)
    # timenow = time.time()
    # #print("Test : ", timenow - time0)        
    # time0 = timenow
    #distributed group
    torch.distributed.get_world_size =  _world_size
    torch.distributed.get_rank = _get_rank
    torch.distributed.get_group_rank = _get_rank

    #cost data transforming time
    torch.Tensor.to = _todevice
    torch.Tensor.cpu = _tocuda

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

