"""
TabPFN regressor wrapper that captures the column (feature) attention map.

PUBLIC API:
  TabPFNWithColAttn.fit(X, y) -> Self
  TabPFNWithColAttn.predict(X) -> np.ndarray
  TabPFNWithColAttn.get_col_attn(reduce) -> np.ndarray
    reduce in {"mean", "last", "per_layer"}

  capture_column_attention(tabpfn_inner_model) -> context manager yielding
    list[tuple[int, torch.Tensor]]; each entry is (layer_idx, attn (B, H, F, F))
    already detached to CPU.

HOW CAPTURE WORKS:
  TabPFN v7's feature attention layer lives at
    PerFeatureEncoderLayer.self_attn_between_features
    (class MultiHeadAttention, defined in
     third-party/TabPFN/src/tabpfn/architectures/base/attention/full_attention.py)
  Its compute path uses torch.nn.functional.scaled_dot_product_attention,
  which does NOT return the softmax weights. To capture them we (per feature-
  attention module only) replace the module's forward with a wrapper that
  monkey-patches torch.nn.functional.scaled_dot_product_attention for the
  duration of the call. Inside the patch we compute the attention by hand
  (einsum + softmax), append the weights to a capture list, and return the
  same output the original SDPA would produce.

  Item-attention modules are untouched because the patch only activates
  inside feature-attention forwards. All patches are restored in finally.

CONFIGURATION KNOBS FOR ALIGNMENT:
  To keep the feature axis of the captured attention aligned with the input
  column order we pass these overrides into TabPFNRegressor:
    n_estimators=1
    inference_config={
      "FEATURE_SHIFT_METHOD": None,   # disables ShuffleFeaturesStep
      "POLYNOMIAL_FEATURES": "no",    # no extra columns
      "FINGERPRINT_FEATURE": False,   # no extra column
      "PREPROCESS_TRANSFORMS": [PreprocessorConfig(name="none",
           categorical_name="none", append_original=False,
           global_transformer_name=None)],  # no SVD / no quantile re-scale
      "OUTLIER_REMOVAL_STD": None,
      "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,),
    }
  See shuffle_features_step.py: when shuffle_method is None the permutation
  is np.arange(F), i.e. identity.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _find_feature_attention_modules(inner_model) -> list[tuple[int, torch.nn.Module]]:
    """
    Walk the loaded TabPFN architecture and return [(layer_idx, feature_attn_module)].

    Supports two architectures:
      * v2.6+ (TabPFNV2p6):
          inner_model.blocks[i].per_sample_attention_between_features
      * v2 legacy (PerFeatureTransformer):
          inner_model.transformer_encoder.layers[i].self_attn_between_features
          (and transformer_decoder.layers[i] if present)
    """
    # v2.6+ first (tabpfn 7.x default).
    blocks = getattr(inner_model, "blocks", None)
    if blocks is not None:
        out: list[tuple[int, torch.nn.Module]] = []
        for i, block in enumerate(blocks):
            attn = getattr(block, "per_sample_attention_between_features", None)
            if attn is not None:
                out.append((i, attn))
        if out:
            return out

    # Legacy PerFeatureTransformer path.
    out = []
    stacks = []
    enc = getattr(inner_model, "transformer_encoder", None)
    if enc is not None:
        stacks.append(enc)
    dec = getattr(inner_model, "transformer_decoder", None)
    if dec is not None:
        stacks.append(dec)
    global_idx = 0
    for stack in stacks:
        layers = getattr(stack, "layers", [])
        for layer in layers:
            attn = getattr(layer, "self_attn_between_features", None)
            if attn is not None:
                out.append((global_idx, attn))
            global_idx += 1
    return out


def _sdpa_capture_factory(captured: list, current_layer_idx: list):
    """
    Build a drop-in replacement for torch.nn.functional.scaled_dot_product_attention
    that records softmax weights and returns the standard attention output.

    q, k, v are shaped (..., H, S, D) or (B*, H, S, D) as callers pass them.
    We handle both by flattening leading dims; the caller re-reshapes the
    output back so shape isn't a concern for them.
    """

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=None, enable_gqa=False):
        if attn_mask is not None or is_causal:
            # TabPFN feature-attn doesn't use these, but fall back safely if it ever does.
            return F._orig_sdpa(  # type: ignore[attr-defined]
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )

        d_k = query.shape[-1]
        s = (1.0 / float(d_k) ** 0.5) if scale is None else scale

        if enable_gqa and key.shape[-3] != query.shape[-3]:
            # Broadcast kv heads to match query heads.
            n_q = query.shape[-3]
            n_kv = key.shape[-3]
            assert n_q % n_kv == 0
            repeat = n_q // n_kv
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)

        logits = torch.matmul(query, key.transpose(-2, -1)) * s
        attn = torch.softmax(logits, dim=-1)
        # Capture after softmax, before dropout, before matmul with V.
        captured.append((current_layer_idx[0], attn.detach().to("cpu")))
        if dropout_p and dropout_p > 0.0:
            attn_for_out = torch.dropout(attn, dropout_p, train=True)
        else:
            attn_for_out = attn
        return torch.matmul(attn_for_out, value)

    return sdpa


@contextmanager
def capture_column_attention(tabpfn_inner_model):
    """
    Context manager: while active, every feature-attention module in the model
    records its softmax weights. Captured as (layer_idx, attn on cpu).

    YIELDS:
      list[tuple[int, torch.Tensor]] - filled in as the model runs.

    NOTE:
      Forwards and the torch.nn.functional SDPA reference are restored in
      finally, even if the wrapped forward raises.
    """
    captured: list[tuple[int, torch.Tensor]] = []
    current_layer_idx: list[int | None] = [None]

    modules = _find_feature_attention_modules(tabpfn_inner_model)
    saved_forwards: list[tuple[torch.nn.Module, object]] = []

    # Keep a reference to the real SDPA so we can fall back on unsupported args.
    if not hasattr(F, "_orig_sdpa"):
        F._orig_sdpa = F.scaled_dot_product_attention  # type: ignore[attr-defined]

    sdpa_patch = _sdpa_capture_factory(captured, current_layer_idx)

    def make_wrapper(layer_idx: int, orig_forward):
        def wrapped(*args, **kwargs):
            prev = current_layer_idx[0]
            current_layer_idx[0] = layer_idx
            saved = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = sdpa_patch
            try:
                return orig_forward(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = saved
                current_layer_idx[0] = prev

        return wrapped

    for layer_idx, mod in modules:
        saved_forwards.append((mod, mod.forward))
        mod.forward = make_wrapper(layer_idx, mod.forward)

    try:
        yield captured
    finally:
        for mod, orig in saved_forwards:
            mod.forward = orig


def _build_inference_config_overrides() -> dict:
    """
    Build an inference_config dict that keeps feature count and order identical
    between input X and the attention matrix. See module docstring for the rationale.
    """
    from tabpfn.preprocessing.configs import PreprocessorConfig

    return {
        "FEATURE_SHIFT_METHOD": None,
        "POLYNOMIAL_FEATURES": "no",
        "FINGERPRINT_FEATURE": False,
        "PREPROCESS_TRANSFORMS": [
            PreprocessorConfig(
                name="none",
                categorical_name="none",
                append_original=False,
                global_transformer_name=None,
            )
        ],
        "OUTLIER_REMOVAL_STD": None,
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,),
    }


class TabPFNWithColAttn:
    def __init__(self, device: str = "cuda", seed: int = 0) -> None:
        """
        ARGS:
          device: "cuda" or "cpu". If "cuda" is requested and unavailable,
            silently falls back to "cpu".
          seed: passed to TabPFNRegressor's random_state.
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda requested but unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device
        self.seed = seed
        self._regressor = None
        self._inner_model = None
        self._last_attn: list[tuple[int, torch.Tensor]] | None = None
        self._n_features: int | None = None

    def _build_regressor(self):
        # Import lazily so importing this module doesn't require tabpfn at top level.
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            n_estimators=1,
            device=self.device,
            random_state=self.seed,
            ignore_pretraining_limits=True,
            inference_config=_build_inference_config_overrides(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNWithColAttn":
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._n_features = X.shape[1]

        self._regressor = self._build_regressor()
        self._regressor.fit(X, y)
        # After fit, models_ is populated. n_estimators=1 → one model.
        self._inner_model = self._regressor.models_[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._regressor is None:
            raise RuntimeError("predict called before fit.")
        X = np.asarray(X)
        with capture_column_attention(self._inner_model) as captured:
            y_pred = self._regressor.predict(X)
        self._last_attn = list(captured)
        return np.asarray(y_pred, dtype=np.float64).ravel()

    def get_col_attn(self, reduce: str = "mean") -> np.ndarray:
        """
        Aggregate the captured per-layer column attention.

        ARGS:
          reduce:
            "mean"      -> per-layer mean over B, H, then mean over layers -> (F, F)
            "last"      -> last layer only, mean over B, H -> (F, F)
            "per_layer" -> per-layer mean over B, H -> (L, F, F)

        RETURNS:
          np.ndarray on CPU, float64.

        NOTES:
          TabPFN internally groups features: if features_per_group=g and there
          are F input features, the transformer attends over G = ceil(F/g)
          groups plus one trailing target column. Raw capture shape is
          (B, H, G+1, G+1).

          To keep the documented (F, F) contract and feature-order alignment
          with MLR's W, we:
            1. Drop the trailing target row/col -> (G, G).
            2. Expand each group entry to a block covering its member features
               so the result is (F, F). Features sharing a group thus share
               identical rows/columns by construction. This is the only
               principled expansion available in v2.6+; v2 ran with
               features_per_group=1 so the expansion was a no-op there.
        """
        if self._last_attn is None:
            raise RuntimeError("get_col_attn called before predict.")
        if len(self._last_attn) == 0:
            raise RuntimeError("No attention was captured during predict.")

        by_layer: dict[int, list[torch.Tensor]] = {}
        for layer_idx, attn in self._last_attn:
            by_layer.setdefault(layer_idx, []).append(attn)

        f = self._n_features if self._n_features is not None else None
        features_per_group = int(getattr(self._inner_model, "features_per_group", 1) or 1)

        layer_ids = sorted(by_layer.keys())
        per_layer = []
        for lid in layer_ids:
            concat = torch.cat(by_layer[lid], dim=0)  # (B_total, H, G+1, G+1)
            mean_bh = concat.mean(dim=(0, 1)).to(torch.float64).numpy()
            # Strip trailing target column/row.
            g = mean_bh.shape[0] - 1
            group_attn = mean_bh[:g, :g]  # (G, G)
            expanded = _expand_group_attn_to_feature(
                group_attn, n_features=f if f is not None else g * features_per_group,
                features_per_group=features_per_group,
            )
            per_layer.append(expanded)
        per_layer_arr = np.stack(per_layer, axis=0)

        if reduce == "per_layer":
            return per_layer_arr
        if reduce == "last":
            return per_layer_arr[-1]
        if reduce == "mean":
            return per_layer_arr.mean(axis=0)
        raise ValueError(f"reduce must be one of mean/last/per_layer, got {reduce!r}")


def _expand_group_attn_to_feature(
    group_attn: np.ndarray, n_features: int, features_per_group: int
) -> np.ndarray:
    """
    Broadcast a (G, G) group-level attention to a (n_features, n_features)
    feature-level matrix by mapping feature i -> group i // features_per_group.

    Features padded into a trailing group (those with i >= G*features_per_group)
    are dropped by clamping the group index.
    """
    g = group_attn.shape[0]
    feature_to_group = np.minimum(np.arange(n_features) // features_per_group, g - 1)
    return group_attn[np.ix_(feature_to_group, feature_to_group)]
