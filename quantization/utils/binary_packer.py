import torch
from typing import Tuple


def binary_packer(tensor: torch.Tensor) -> torch.Tensor:
    """  
    Pack a {-1,+1} int8 matrix into row-major 32-bit words.  

    This is a simplified version that only handles single tensor packing with "lsb_first" method
    
    Parameters  
    ----------  
    tensor : torch.Tensor[int8]   (n_rows, n_cols)  
        Matrix to pack. Values must be exactly -1 or +1.

    Returns  
    -------  
    torch.Tensor[int32]  
        Packed tensor of shape ``(n_rows, words_per_row)``.

    Raises  
    ------  
    TypeError  
        * Input tensor is not ``int8``.
    """
    # Validate tensor dtype
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")

    # Get tensor dimensions
    n_rows, n_cols = tensor.shape
    words_per_row: int = (n_cols + 31) // 32
    pad: int = words_per_row * 32 - n_cols
    device = tensor.device

    # Pad with +1 (which becomes bit-0 after the +1 > 0 / -1 > 1 conversion)
    if pad:
        pad_tensor = torch.ones((n_rows, pad), dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    # Convert to bits: +1 > 0 , -1 > 1   (uint8)
    assert tensor.ndim == 2 and tensor.size(1) % 32 == 0
    bits = ((1 - tensor) // 2).int()
    bits = bits.reshape(tensor.size(0), tensor.size(1) // 32, 32)

    # Build per-bit weights for lsb_first ordering
    wts = (2**torch.arange(32, dtype=torch.int32, device=device)).int()

    # Aggregate bits into 32-bit words
    packed = (bits * wts).sum(dim=2, dtype=torch.int32)  # (row, word)
    return packed.contiguous()


def binary_unpacker(packed_tensor: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """  
    Unpack a tensor from packed format back to {-1,+1} int8 matrix.

    Parameters  
    ----------  
    packed_tensor : torch.Tensor[int32]   (n_rows, words_per_row)  
        Packed tensor to unpack. Must be a 2D tensor with the expected shape.
    original_shape : Tuple[int, int]  
        The original (n_rows, n_cols) shape of the tensor before packing.

    Returns  
    -------  
    torch.Tensor[int8]  
        Unpacked tensor of shape ``original_shape`` with values {-1,+1}.

    Raises  
    ------  
    ValueError  
        * Packed tensor format is not supported (not 2D or incorrect shape).
    """
    # Handle different packing formats
    if packed_tensor.dim() == 2:  # Simple 1-bit packing case
        n_rows, n_cols = original_shape
        words_per_row = (n_cols + 31) // 32
        if packed_tensor.shape == (n_rows, words_per_row):
            # Extract all words' bits in one shot (lsb_first), then trim padding
            shifts = torch.arange(32, dtype=torch.int32, device=packed_tensor.device)
            unpacked = ((packed_tensor.unsqueeze(-1) >> shifts) & 1).to(torch.int8)
            unpacked = unpacked.reshape(n_rows, words_per_row * 32)[:, :n_cols]
            return (1 - 2 * unpacked).contiguous()  # int8 in {-1,+1}

    # Raise error for unsupported cases
    raise ValueError(f"Unsupported packed tensor format. Expected 2D tensor with shape "
                     f"(n_rows, words_per_row), got {packed_tensor.shape} with {packed_tensor.dim()} dimensions.")
