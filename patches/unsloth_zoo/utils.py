# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "Version",
    "_get_dtype",
    "is_main_process",
    "is_distributed",
    "distributed_function",
]

from packaging.version import Version as TrueVersion
import torch

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass
pass


__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}
def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            try: dtype = eval(f"torch.{dtype.lower()}")
            except: pass
        if type(dtype) is torch.dtype: return dtype
    return None
pass


def is_main_process():
    is_initialized = torch.distributed.is_initialized()
    return (not is_initialized) or (is_initialized and torch.distributed.get_rank() == 0)
pass


def is_distributed():
    return torch.distributed.is_initialized()
pass


def distributed_function(n=1, function=None, *args, **kwargs):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Debug: sync and log `n` across all ranks
        n_tensor = torch.tensor([n], device="cuda" if torch.cuda.is_available() else "cpu")
        torch.distributed.broadcast(n_tensor, src=0)
        n = n_tensor.item()

        if rank == 0:
            result = function(*args, **kwargs)

            if not isinstance(result, (list, tuple)):
                object_list = [result] * n
            else:
                object_list = list(result)
                if len(object_list) != n:
                    raise ValueError(f"[Rank 0] Expected {n} elements, but got {len(object_list)}: {result}")
        else:
            object_list = [None for _ in range(n)]

        print(f"[Rank {rank}] Broadcasting object_list (len={len(object_list)}): {object_list}")

        torch.distributed.broadcast_object_list(
            object_list,
            src=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        return object_list if n != 1 else object_list[0]

    else:
        result = function(*args, **kwargs)
        if n == 1:
            return result
        if isinstance(result, (list, tuple)) and len(result) == n:
            return list(result)
        else:
            raise ValueError(f"[Non-distributed] Expected {n} elements, but got: {result}")

pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
