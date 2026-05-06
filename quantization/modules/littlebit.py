import os
import torch
import torch.nn as nn

from quantization.utils.binary_packer import binary_packer


class LittleBitLinear(nn.Module):
    def __quant_convert__(
        self,
        do_train: bool,
        quant_func: torch.autograd.Function,
        *,
        split_dim: int = 1024,
        eff_bit: float | None = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        use_itq: bool = False,
        itq_n_iter: int = 50,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.residual = residual
        self.use_itq = use_itq
        self.itq_n_iter = itq_n_iter

        # Flag to track if weights are binarized to int8 for inference
        self._binarized = False

        eff_bit_target = eff_bit

        a, b = self.in_features, self.out_features

        split_calc_float = self._estimate_split_dim(a, b, eff_bit_target, residual)

        if split_calc_float:
            split_calc_float *= ratio_factor

        final_split_dim = self._finalize_split_dim(split_calc_float, split_dim, min_split_dim)
        self.split_dim = final_split_dim

        eff_bit_actual = self._compute_eff_bits(a, b, final_split_dim, residual)
        self.register_buffer("_eff_bit_target", torch.tensor(-1.0 if eff_bit_target is None else float(eff_bit_target)))
        self.register_buffer("_split_dim_final", torch.tensor(final_split_dim))
        self.register_buffer("_eff_bit_actual", torch.tensor(eff_bit_actual))

        if self.do_train and hasattr(self, 'weight') and self.weight is not None:
            self._initialize_parameters()
        else:
            self._initialize_empty_parameters()

    @staticmethod
    def _estimate_split_dim(a, b, eff_bit_target, residual) -> float | None:
        """Estimate the initial (float) value of split_dim based on bit target."""
        if eff_bit_target is None or a * b == 0:
            return None

        base = a + b + 16
        if residual:
            numerator = a * b * eff_bit_target - 32 * (a + b)
            denominator = 2 * base
        else:
            numerator = a * b * eff_bit_target - 16 * (a + b)
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
    ) -> int:
        """Round down to nearest multiple of 8 and apply minimum fallback."""
        # Use default if no split estimate is available
        cand = split_float if split_float is not None else split_default
        cand = int(cand) if cand is not None else 0

        # Round down to a multiple of 8
        cand = (cand // 8) * 8
        if cand == 0:
            cand = min_split_dim

        return max(cand, min_split_dim)

    @staticmethod
    def _compute_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
        """Calculate the actual effective bits used based on configuration."""
        if a * b == 0:
            return float("inf")

        if residual:
            num = s * 2 * (a + b + 16) + 32 * (a + b)
        else:
            num = s * (a + b + 16) + 16 * (a + b)
        return num / (a * b)

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_features)
        hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        # Compute main forward pass
        y = self._compute_forward(x, self.V, self.U, self.v2, self.v1, self.u2, self.u1)

        if self.residual:
            # Compute residual forward pass
            res = self._compute_forward(x, self.V_R, self.U_R, self.v2_R, self.v1_R, self.u2_R, self.u1_R)
            y = y + res

        if self.bias is not None:
            y += self.bias
        y = y.reshape(hidden_output_dim)
        return y

    def _compute_forward(self, x, V, U, v2, v1, u2, u1):
        """Helper method to compute the forward pass for both main and residual components."""
        Vq = self.quantize(V.to(x.dtype))
        Uq = self.quantize(U.to(x.dtype))
        v1u2 = v1 * u2

        # ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
        return ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

    def quantize(self, x):
        # If weights are already binarized, return them directly
        if self._binarized:
            return x
        # Otherwise, apply quantization function
        return self.quant_func(x)

    def extra_repr(self):
        params = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
            "split_dim": self._split_dim_final,
            "eff_bit_target": f"{self.eff_bit_target:.4f}" if self.eff_bit_target is not None else "N/A",
            "eff_bit_actual": f"{self.eff_bit_actual:.4f}",
            "residual": self.residual,
            "total_bit_usage": f"{self.total_bit_usage:.0f}"
        }

        return ", ".join(f"{key}={value}" for key, value in params.items())

    def _initialize_empty_parameters(self):
        """Initialize with empty parameters for memory efficiency during inference"""
        dtype = torch.bfloat16  # temporary dtype, actual values loaded from state_dict
        device = "meta"  # use meta device to prevent actual memory allocation

        # Helper function to create parameters with consistent settings
        def create_param(*shape):
            return nn.Parameter(torch.empty(*shape, device=device, dtype=dtype), requires_grad=self.do_train)

        # Initialize main parameters
        self.U = create_param(self.out_features, self.split_dim)
        self.V = create_param(self.split_dim, self.in_features)
        self.u1 = create_param(1, self.out_features)
        self.u2 = create_param(1, self.split_dim)
        self.v1 = create_param(1, self.split_dim)
        self.v2 = create_param(1, self.in_features)

        if self.residual:
            # Initialize residual parameters
            self.U_R = create_param(self.out_features, self.split_dim)
            self.V_R = create_param(self.split_dim, self.in_features)
            self.u1_R = create_param(1, self.out_features)
            self.u2_R = create_param(1, self.split_dim)
            self.v1_R = create_param(1, self.split_dim)
            self.v2_R = create_param(1, self.in_features)

        # Delete original weight
        if hasattr(self, 'weight'):
            del self.weight
        self.register_parameter('weight', None)

    def _initialize_parameters(self):
        # Only perform actual decomposition when `do_train` is True.
        W = self.weight.data.float() if self.do_train and self.weight is not None else None

        U, V, u1, u2, v1, v2 = self._decompose_matrix(W)

        # Helper function to create parameters with consistent settings
        def create_param(tensor):
            return nn.Parameter(tensor, requires_grad=self.do_train)

        # Initialize main parameters
        self.U = create_param(U)
        self.V = create_param(V)
        self.v1 = create_param(v1)
        self.v2 = create_param(v2)
        self.u1 = create_param(u1)
        self.u2 = create_param(u2)

        if self.residual:
            residual_W = None
            if self.do_train:
                # Offload the heavy W_approx matrix multiplication to GPU to prevent CPU bottleneck
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                calc_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else self.weight.device

                # Temporarily move components to calc_device
                U_g = self.quantize(U).to(calc_device)
                V_g = self.quantize(V).to(calc_device)
                u1_g, u2_g = u1.to(calc_device), u2.to(calc_device)
                v1_g, v2_g = v1.to(calc_device), v2.to(calc_device)

                # Calculate approximation using decomposed matrices on GPU
                W_approx_g = (U_g * (u1_g.t() @ u2_g)) @ (V_g * (v1_g.t() @ v2_g))

                # Move the result back to the original device (CPU) for subtraction
                residual_W = self.weight.data.float() - W_approx_g.to(self.weight.device)

                # Clean up GPU memory
                del U_g, V_g, u1_g, u2_g, v1_g, v2_g, W_approx_g

            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(residual_W)

            # Initialize residual parameters
            self.U_R = create_param(U_R)
            self.V_R = create_param(V_R)
            self.v1_R = create_param(v1_R)
            self.v2_R = create_param(v2_R)
            self.u1_R = create_param(u1_R)
            self.u2_R = create_param(u2_R)

        # After decomposition, the original weight is no longer needed, so set it to None
        self.register_parameter('weight', None)
        self._binarized = False

    def _compute_itq_rotation(self, X, n_iter=20):
        """
        Finds optimal Rotation R that aligns data to the binary hypercube.
        Objective: min || B - X @ R ||_F^2  s.t. B = sign(X @ R)
        """
        with torch.no_grad():
            _, dim = X.shape
            device = X.device

            X_f = X.float()

            R = torch.empty((dim, dim), device=device, dtype=torch.float32)
            torch.nn.init.orthogonal_(R)

            for _ in range(n_iter):
                Z = X_f @ R
                B = torch.sign(Z)

                M = B.t() @ X_f
                U_p, _, Vt_p = torch.linalg.svd(M, full_matrices=False)

                R = Vt_p.t() @ U_p.t()

            return R.to(X.dtype)

    def _decompose_matrix(self, X=None):
        """
        Computes a low-rank decomposition of matrix X via SVD.
        Then aligns the factors with joint ITQ and applies an extra SVD
        (on the absolute value) to each of the two factors
        for additional factorization into (vector1, vector2) pairs.
        Returns:
            U, V: The low-rank factors of the original matrix.
            u1, u2: The pair from further decomposition on U.
            v1, v2: The pair from further decomposition on V.
        """
        if self.do_train:
            assert X is not None, "Weight matrix X must be provided for training initialization."
            assert X.shape[0] == self.out_features
            assert X.shape[1] == self.in_features

            original_device = X.device

            # Get LOCAL_RANK for DeepSpeed/DDP compatibility. Default to 0 for single GPU.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            calc_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else original_device

            # Move tensor to the target device for faster computation
            X_calc = X.to(calc_device)

            # Use svd_lowrank instead of linalg.svd to drastically speed up initialization
            U_t, S_t, V_t = torch.svd_lowrank(X_calc, q=self.split_dim)
            Vh_t = V_t.t()

            sqrt_S = torch.sqrt(torch.diag(S_t))[:, :self.split_dim]

            U = (U_t @ sqrt_S).contiguous()
            V = (sqrt_S.t() @ Vh_t).contiguous()

            with torch.no_grad():
                if self.use_itq:
                    X_combined = torch.cat([U, V.t()], dim=0)
                    R = self._compute_itq_rotation(X_combined, n_iter=self.itq_n_iter)
                    U = (U @ R).contiguous()
                    V = (R.t() @ V).contiguous()
                    del X_combined, R

            v1, v2 = self._rank_one_decompose(torch.abs(V), calc_device=calc_device)
            u1, u2 = self._rank_one_decompose(torch.abs(U), calc_device=calc_device)

            dtype = X.dtype
            # Safely move the computed tensors back to the original device (e.g., CPU for ZeRO-3)
            U = U.to(device=original_device, dtype=dtype)
            V = V.to(device=original_device, dtype=dtype)
            v1 = v1.to(device=original_device, dtype=dtype)
            v2 = v2.to(device=original_device, dtype=dtype)
            u1 = u1.to(device=original_device, dtype=dtype)
            u2 = u2.to(device=original_device, dtype=dtype)

            # Explicitly delete temporary GPU tensors to prevent OOM
            del X_calc, U_t, S_t, V_t, Vh_t

        else:
            U = torch.empty(self.out_features, self.split_dim)
            V = torch.empty(self.split_dim, self.in_features)
            u1 = torch.empty(1, self.out_features)
            u2 = torch.empty(1, self.split_dim)
            v1 = torch.empty(1, self.split_dim)
            v2 = torch.empty(1, self.in_features)
        return U, V, u1, u2, v1, v2

    def _rank_one_decompose(self, X, calc_device=None):
        """
        Perform rank-one decomposition on matrix X via SVD and return two vectors.
        """
        original_device = X.device
        if calc_device is None:
            # Apply LOCAL_RANK logic to ensure strict device placement
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            calc_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else original_device

        X_calc = X.to(calc_device)

        # Use svd_lowrank with q=1 for ultra-fast rank-one decomposition
        U, S, V = torch.svd_lowrank(X_calc, q=1)
        Vh = V.t()

        sqrt_S0 = torch.sqrt(S[0])
        u_component = (U[:, :1] * sqrt_S0).t().contiguous()
        v_component = (sqrt_S0 * Vh[:1, :]).contiguous()
        return u_component, v_component

    def pack_weights(self, *args, **kwargs):
        """
        Pack binary weights. Shapes are converted to tensors to ensure
        the state_dict contains only tensors.
        """
        packed_data = {}

        # Helper function to binarize and pack a parameter
        def pack_param(param, name):
            param_bin = self.quantize(param.data).to(torch.int8)
            packed_data[f'{name}_packed'] = binary_packer(param_bin)
            packed_data[f'{name}_shape'] = torch.tensor(param.shape, dtype=torch.long)

        # Pack main parameters
        pack_param(self.U, 'U')
        pack_param(self.V, 'V')

        if self.residual:
            # Pack residual parameters
            pack_param(self.U_R, 'U_R')
            pack_param(self.V_R, 'V_R')

        return packed_data

    def state_dict(self, *args, **kwargs):
        """Always return the state_dict in a binarized & packed format."""
        prefix = kwargs.get('prefix', '')
        state = super().state_dict(*args, **kwargs)

        keys_to_remove = [k for k in state.keys() if k.startswith(prefix + 'U') or k.startswith(prefix + 'V')]
        for k in keys_to_remove:
            if k in state:
                del state[k]

        packed_weights = self.pack_weights()
        for k, v in packed_weights.items():
            state[prefix + k] = v

        return state

    @property
    def eff_bit_target(self):
        v = self._eff_bit_target.item()
        return None if v < 0 else v

    @property
    def eff_bit_actual(self):
        return self._eff_bit_actual.item()

    @property
    def split_dim_used(self):
        return int(self._split_dim.item())

    @property
    def total_bit_usage(self):
        return self.eff_bit_actual * self.in_features * self.out_features
