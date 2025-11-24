import torch
import torch.nn as nn


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
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.residual = residual

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

        self._initialize_parameters()

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

        Vq = self.quantize(self.V)
        Uq = self.quantize(self.U)
        v2 = self.v2
        v1u2 = self.v1 * self.u2
        u1 = self.u1

        # ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
        y = ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

        if self.residual:
            Vq_R = self.quantize(self.V_R)
            Uq_R = self.quantize(self.U_R)
            v2_R = self.v2_R
            v1u2_R = self.v1_R * self.u2_R
            u1_R = self.u1_R

            res = ((((x * v2_R) @ Vq_R.t()) * v1u2_R) @ Uq_R.t()) * u1_R
            y = y + res

        if self.bias is not None:
            y += self.bias
        y = y.reshape(hidden_output_dim)
        return y

    def quantize(self, x):
        return self.quant_func().apply(x)

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

    def _initialize_parameters(self):
        W = self.weight.data.float() if self.do_train else None
        U, V, u1, u2, v1, v2 = self._decompose_matrix(W)

        self.register_parameter('U', nn.Parameter(U))
        self.register_parameter('V', nn.Parameter(V))
        self.register_parameter('v1', nn.Parameter(v1))
        self.register_parameter('v2', nn.Parameter(v2))
        self.register_parameter('u1', nn.Parameter(u1))
        self.register_parameter('u2', nn.Parameter(u2))

        if self.residual:
            residual_W = None
            if self.do_train:
                UV_approx = (U.sign() * (u1.t() @ u2)) @ (V.sign() * (v1.t() @ v2))
                residual_W = self.weight.data.float() - UV_approx
            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(residual_W)

            self.register_parameter('U_R', nn.Parameter(U_R))
            self.register_parameter('V_R', nn.Parameter(V_R))
            self.register_parameter('v1_R', nn.Parameter(v1_R))
            self.register_parameter('v2_R', nn.Parameter(v2_R))
            self.register_parameter('u1_R', nn.Parameter(u1_R))
            self.register_parameter('u2_R', nn.Parameter(u2_R))

        self.register_parameter('weight', None)

    def _decompose_matrix(self, X=None):
        """
        Computes a low-rank decomposition of matrix X via SVD.
        Then applies an extra SVD (on the absolute value) to each of the two factors
        for additional factorization into (vector1, vector2) pairs.
        Returns:
            U, V: The low-rank factors of the original matrix.
            u1, u2: The pair from further decomposition on U.
            v1, v2: The pair from further decomposition on V.
        """
        if self.do_train:
            assert X.shape[0] == self.out_features
            assert X.shape[1] == self.in_features
            U_t, S_t, Vh_t = torch.linalg.svd(X, full_matrices=False)
            sqrt_S = torch.sqrt(torch.diag(S_t))[:, :self.split_dim]
            U = (U_t @ sqrt_S).contiguous()
            V = (sqrt_S.t() @ Vh_t).contiguous()

            v1, v2 = self._rank_one_decompose(torch.abs(V))
            u1, u2 = self._rank_one_decompose(torch.abs(U))

            dtype = torch.bfloat16
            U, V = U.to(dtype), V.to(dtype)
            v1, v2 = v1.to(dtype), v2.to(dtype)
            u1, u2 = u1.to(dtype), u2.to(dtype)
        else:
            U = torch.empty(self.out_features, self.split_dim)
            V = torch.empty(self.split_dim, self.in_features)
            u1 = torch.empty(1, self.out_features)
            u2 = torch.empty(1, self.split_dim)
            v1 = torch.empty(1, self.split_dim)
            v2 = torch.empty(1, self.in_features)
        return U, V, u1, u2, v1, v2

    def _rank_one_decompose(self, X):
        """
        Perform rank-one decomposition on matrix X via SVD and return two vectors.
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        sqrt_S0 = torch.sqrt(S[0])
        u_component = (U[:, :1] * sqrt_S0).t().contiguous()
        v_component = (sqrt_S0 * Vh[:1, :]).contiguous()
        return u_component, v_component

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
