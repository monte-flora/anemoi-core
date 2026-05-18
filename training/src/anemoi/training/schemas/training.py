# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from enum import Enum
from functools import partial
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import allowed_values


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    "Gradient clipping value."
    algorithm: Annotated[str, AfterValidator(partial(allowed_values, values=["value", "norm"]))] = Field(
        example="value",
    )
    "The gradient clipping algorithm to use"


class SWA(BaseModel):
    """Stochastic weight averaging configuration.

    See https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    """

    enabled: bool = Field(example=False)
    "Enable stochastic weight averaging."
    lr: NonNegativeFloat = Field(example=1.0e-4)
    "Learning rate for SWA."


class EWA(BaseModel):
    """Exponential weight averaging configuration.

    See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.WeightAveraging.html
    """

    enabled: bool = Field(default=False)
    "Enable exponential weight averaging."
    ema_decay: float = Field(default=0.999, ge=0.0, le=1.0)
    "Decay factor for exponential moving average. Higher values (closer to 1.0) give more weight to recent weights."


class Rollout(BaseModel):
    """Rollout configuration."""

    start: PositiveInt = Field(example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(example=0)
    "Number of epochs to increment the rollout."
    max: PositiveInt = Field(example=1)
    "Maximum number of rollouts."


class LR(BaseModel):
    """Learning rate configuration.

    ``semantics`` controls how ``rate`` and ``min`` are interpreted relative
    to the actual optimizer LR the model sees during training:

    - ``"per_rank_legacy"`` (default): legacy anemoi behaviour. ``rate`` is
      multiplied by ``num_nodes × num_gpus_per_node / num_gpus_per_model``
      to obtain the optimizer peak, but ``min`` is passed through literally
      (no multiplier). Kept as the default so existing configs continue to
      reproduce historical results; the asymmetry is a known footgun.
    - ``"per_rank"``: same multiplier applied to ``rate`` AND ``min``. The
      asymmetry is gone — cosine sweep span is now hardware-independent.
      Recommended when migrating a per-rank config away from the legacy
      behaviour without rewriting the literal numbers.
    - ``"global"``: ``rate`` and ``min`` are the literal values the
      optimizer will use, regardless of GPU count. Cleanest semantics.
      Recommended for all new configs.

    Changes in per-gpu batch_size still warrant rescaling: in ``"global"``
    semantics, just multiply ``rate`` and ``min`` by the batch ratio.
    """

    semantics: Literal["per_rank_legacy", "per_rank", "global"] = "per_rank_legacy"
    "How rate/min are interpreted relative to optimizer LR. See class docstring."
    rate: NonNegativeFloat = Field(example=0.625e-4)
    "Initial learning rate, interpreted per ``semantics``."
    iterations: NonNegativeInt = Field(example=300000)
    "Number of iterations."
    min: NonNegativeFloat = Field(example=3e-7)
    "Minimum learning rate, interpreted per ``semantics``."
    warmup: NonNegativeInt = Field(example=1000)
    "Number of warm up iteration. Default to 1000."


class OptimizerSchema(PydanticBaseModel):
    """Choosing the PydanticBaseModel to allow extra inputs."""

    model_config = ConfigDict(extra="allow")

    target_: str = Field(..., alias="_target_")
    """Full path to the optimizer class, e.g. `torch.optim.AdamW`."""


class ExplicitTimes(BaseModel):
    """Time indices for input and output.

    Starts at index 0. Input and output can overlap.
    """

    input: list[NonNegativeInt] = Field(examples=[0, 1])
    "Input time indices."
    target: list[NonNegativeInt] = Field(examples=[2])
    "Target time indices."


class TargetForcing(BaseModel):
    """Forcing parameters for target output times.

    Extra forcing parameters to use as input to distinguish between different target times.
    """

    data: list[str] = Field(examples=["insolation"])
    "List of forcing parameters to use as input to the model at the interpolated step."
    time_fraction: bool = Field(example=True)
    "Use target time as a fraction between input boundary times as input."


class LossScalingSchema(BaseModel):
    default: int = 1
    "Default scaling value applied to the variables loss. Default to 1."
    pl: dict[str, NonNegativeFloat]
    "Scaling value associated to each pressure level variable loss."
    sfc: dict[str, NonNegativeFloat]
    "Scaling value associated to each surface variable loss."


class GeneralVariableLossScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.GeneralVariableLossScaler"] = Field(..., alias="_target_")
    weights: dict[str, float]
    "Weight of each variable."  # Check keys (variables) are read ???


class VariableMaskingScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.VariableMaskingLossScaler"] = Field(..., alias="_target_")
    variables: list[str] = Field(defaultexample=["tp"])
    "Variables to compute the loss over."
    invert: bool = Field(examples=False)
    "Flag to invert the variable mask."


class NaNMaskScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.NaNMaskScaler"] = Field(..., alias="_target_")
    use_processors_tendencies: bool = Field(default=False)
    "Flag to include processors for tendencies when building the loss mask."


class TendencyScalerTargets(str, Enum):
    stdev = "anemoi.training.losses.scalers.StdevTendencyScaler"
    var = "anemoi.training.losses.scalers.VarTendencyScaler"


class TendencyScalerSchema(BaseModel):
    target_: TendencyScalerTargets = Field(
        example="anemoi.training.losses.scalers.StdevTendencyScaler",
        alias="_target_",
    )


class VariableLevelScalerTargets(str, Enum):
    relu_scaler = "anemoi.training.losses.scalers.ReluVariableLevelScaler"
    linear_scaler = "anemoi.training.losses.scalers.LinearVariableLevelScaler"
    polynomial_sclaer = "anemoi.training.losses.scalers.PolynomialVariableLevelScaler"
    no_scaler = "anemoi.training.losses.scalers.NoVariableLevelScaler"
    model_level_scaler = "anemoi.training.losses.scalers.ModelLevelReluVariableLevelScaler"
    level_average_scaler = "anemoi.training.losses.scalers.LevelAverageScaler"


class VariableLevelScalerSchema(BaseModel):
    target_: VariableLevelScalerTargets = Field(
        example="anemoi.training.losses.scalers.ReluVariableLevelScaler",
        alias="_target_",
    )
    group: str = Field(example="pl")
    "Group of variables to scale."
    slope: float = Field(default=0.0, example=1.0)
    "Slope of scaling function (unused by NoVariableLevelScaler / LevelAverageScaler)."
    y_intercept: float = Field(default=1.0, example=0.001)
    "Y-axis shift of scaling function (unused by NoVariableLevelScaler / LevelAverageScaler)."


class GraphNodeAttributeScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.GraphNodeAttributeScaler"] = Field(..., alias="_target_")
    nodes_name: str = Field(example="data")
    "Name of the nodes to take the attribute from."
    nodes_attribute_name: str = Field(example="area_weight")
    "Name of the node attribute to return."
    norm: Literal["unit-max", "unit-sum"] | None = Field(example="unit-sum")
    "Normalisation method applied to the node attribute."


class ReweightedGraphNodeAttributeScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.ReweightedGraphNodeAttributeScaler"] = Field(
        ...,
        alias="_target_",
    )
    nodes_name: str = Field(example="data")
    "Name of the nodes to take the attribute from."
    nodes_attribute_name: str = Field(example="area_weight")
    "Name of the node attribute to return."
    scaling_mask_attribute_name: str = Field(example="cutout_mask")
    "Name of the node attribute to use as a mask to reweight the reference values."
    weight_frac_of_total: float = Field(example=0.5)
    "Fraction of total weight to assign to nodes within the scaling mask. The remaining weight is distributed among "
    "nodes outside the mask."
    norm: Literal["unit-max", "unit-sum"] | None = Field(example="unit-sum")
    "Normalisation method applied to the node attribute."


ScalerSchema = (
    GeneralVariableLossScalerSchema
    | VariableLevelScalerSchema
    | VariableMaskingScalerSchema
    | TendencyScalerSchema
    | NaNMaskScalerSchema
    | GraphNodeAttributeScalerSchema
    | ReweightedGraphNodeAttributeScalerSchema
)


class ImplementedLossesUsingBaseLossSchema(str, Enum):
    kcrps = "anemoi.training.losses.kcrps.KernelCRPS"
    afkcrps = "anemoi.training.losses.kcrps.AlmostFairKernelCRPS"
    rmse = "anemoi.training.losses.RMSELoss"
    mse = "anemoi.training.losses.MSELoss"
    weighted_mse = "anemoi.training.losses.WeightedMSELoss"
    mae = "anemoi.training.losses.MAELoss"
    logcosh = "anemoi.training.losses.LogCoshLoss"
    huber = "anemoi.training.losses.HuberLoss"
    rmse_norm = "anemoi.training.losses.RMSELossNormalized"
    combined = "anemoi.training.losses.combined.CombinedLoss"
    graphcast_combined = "anemoi.training.losses.graphcast_combined.GraphCastCombinedLoss"
    graphcast_full = "anemoi.training.losses.graphcast_full.GraphCastFullLoss"
    horizontal_gradient = "anemoi.training.losses.horizontal_gradient.HorizontalGradientLoss"
    graphcast_wind = "anemoi.training.losses.graphcast_wind.GraphCastWindAwareLoss"
    graphcast_mse = "anemoi.training.losses.GraphCastMSELoss"
    graphcast_huber = "anemoi.training.losses.GraphCastHuberLoss"
    graphcast_logcosh = "anemoi.training.losses.GraphCastLogCoshLoss"
    graphcast_clipped_mse = "anemoi.training.losses.GraphCastClippedMSELoss"
    graphcast_pseudo_huber = "anemoi.training.losses.GraphCastPseudoHuberLoss"
    graphcast_gaussian_nll = "anemoi.training.losses.GraphCastGaussianNLLLoss"
    graphcast_crps = "anemoi.training.losses.GraphCastCRPSLoss"
    fcl = "anemoi.training.losses.spectral.FourierCorrelationLoss"
    lsd = "anemoi.training.losses.spectral.LogSpectralDistance"


class BaseLossSchema(BaseModel):
    target_: ImplementedLossesUsingBaseLossSchema = Field(..., alias="_target_")
    "Loss function object from anemoi.training.losses."
    scalers: list[str] = Field(example=["variable"])  # TODO(Mario): Validate scalers are defined
    "Scalars to include in loss calculation"
    ignore_nans: bool = False
    "Allow nans in the loss and apply methods ignoring nans for measuring the loss."


class KernelCRPSSchema(BaseLossSchema):
    fair: bool = True
    "Calculate a 'fair' (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1"


class AlmostFairKernelCRPSSchema(BaseLossSchema):
    alpha: float = 1.0
    """Factor for linear combination of fair (unbiased, ensemble variance component
    weighted by (ens-size-1)^-1) and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)"""
    no_autocast: bool = True
    "Deactivate autocast for the kernel CRPS calculation"


class MultiScaleLossSchema(BaseModel):
    target_: Literal["anemoi.training.losses.MultiscaleLossWrapper"] = Field(..., alias="_target_")
    per_scale_loss: AlmostFairKernelCRPSSchema | KernelCRPSSchema
    weights: list[float]
    keep_batch_sharded: bool
    loss_matrices_path: str
    loss_matrices: list[str | None]

    @field_validator("weights")
    @classmethod
    def validate_weights_length(cls, v: list[float], info: Any) -> list[float]:
        if "loss_matrices" in info.data:
            assert len(v) == len(info.data["loss_matrices"]), "weights must have same length as loss_matrices"
        return v


class HuberLossSchema(BaseLossSchema):
    delta: float = 1.0
    "Threshold for Huber loss."


class SpectralLossSchema(BaseLossSchema):
    """Spectral loss class (upstream consolidated module)."""

    transform: Literal["fft2d", "sht"] = Field(..., example="fft2d")
    """Type of spectral transform to use."""

    class Config(BaseModel.Config):
        """Override to allow extra parameters for spectral transforms."""

        extra = "allow"


class LogFFT2DistanceSchema(BaseLossSchema):
    """Schema for LogFFT2Distance (log-spectral distance) loss.

    Inherits BaseLossSchema so it's usable in any LossSchemas context including
    as an element of CombinedLossSchema.losses.
    """

    target_: Literal["anemoi.training.losses.spatial.LogFFT2Distance"] = Field(..., alias="_target_")
    "Spatial log-spectral distance loss from anemoi.training.losses.spatial."
    x_dim: int = Field(..., example=246)
    "X dimension of the 2D grid (must satisfy x_dim * y_dim == grid size)."
    y_dim: int = Field(..., example=246)
    "Y dimension of the 2D grid."


class FourierCorrelationLossSchema(BaseLossSchema):
    """Schema for FourierCorrelationLoss. Inherits BaseLossSchema for compatibility."""

    target_: Literal["anemoi.training.losses.spatial.FourierCorrelationLoss"] = Field(..., alias="_target_")
    x_dim: int = Field(..., example=246)
    y_dim: int = Field(..., example=246)


class SpectralAmplitudeLossSchema(BaseLossSchema):
    """Schema for SpectralAmplitudeLoss (MSH) — 2D Cartesian amplitude-only spectral loss.

    Inherits BaseLossSchema so it's usable in any LossSchemas context including
    as an element of CombinedLossSchema.losses.
    """

    target_: Literal[
        "anemoi.training.losses.graphcast_msh.GraphCastMSHLoss",
        "anemoi.training.losses.graphcast_msh.SpectralAmplitudeLoss",
        "anemoi.training.losses.graphcast_msh.MSHLoss",
    ] = Field(..., alias="_target_")
    "Modified Spherical Harmonic / spectral amplitude loss."
    x_dim: int = Field(..., example=246)
    "X dimension of the 2D grid (must satisfy x_dim * y_dim == grid size)."
    y_dim: int = Field(..., example=246)
    "Y dimension of the 2D grid."
    high_k_weight_exponent: float = 0.0
    "Exponent for optional (k / k_max) ** exponent per-bin weighting. 0.0 = uniform."


class SpatialGradientLossSchema(BaseLossSchema):
    """Schema for SpatialGradientLoss — 2D Cartesian finite-difference gradient MSE.

    Inherits BaseLossSchema so it's usable in any LossSchemas context including
    as an element of CombinedLossSchema.losses.
    """

    target_: Literal[
        "anemoi.training.losses.gradient.SpatialGradientLoss",
        "anemoi.training.losses.gradient.SobelLoss",
    ] = Field(..., alias="_target_")
    "Spatial gradient (edge/sharpness) loss on a 2D Cartesian LAM grid."
    x_dim: int = Field(..., example=246)
    "X dimension of the 2D grid (must satisfy x_dim * y_dim == grid size)."
    y_dim: int = Field(..., example=246)
    "Y dimension of the 2D grid."


class GraphCastMSELossSchema(BaseLossSchema):
    """Schema for GraphCast-style MSE loss with sample weighting."""

    sample_weighting: bool = False
    "If True, downweight samples with extreme target values."
    sample_weight_threshold: float = 10.0
    "Target magnitude threshold for weighting."
    sample_weight_min: float = 0.01
    "Minimum weight for extreme samples."


class GraphCastHuberLossSchema(BaseLossSchema):
    """Schema for GraphCast-style Huber loss with sample weighting."""

    delta: float = 1.0
    "Threshold for switching from quadratic to linear loss."
    sample_weighting: bool = False
    "If True, downweight samples with extreme target values."
    sample_weight_threshold: float = 10.0
    "Target magnitude threshold for weighting."
    sample_weight_min: float = 0.01
    "Minimum weight for extreme samples."


class GraphCastLogCoshLossSchema(BaseLossSchema):
    """Schema for GraphCast-style LogCosh loss with sample weighting."""

    sample_weighting: bool = False
    "If True, downweight samples with extreme target values."
    sample_weight_threshold: float = 10.0
    "Target magnitude threshold for weighting."
    sample_weight_min: float = 0.01
    "Minimum weight for extreme samples."


class GraphCastClippedMSELossSchema(BaseLossSchema):
    """Schema for GraphCast-style MSE loss with per-element clipping."""

    clip_value: float = 100.0
    "Maximum squared error per element. Prevents gradient explosion from extreme errors."
    log_clipping_stats: bool = True
    "If True, log statistics about clipping behavior."
    log_per_variable_loss: bool = True
    "If True, log loss per variable group."
    log_interval: int = 100
    "Log statistics every N forward passes."


class GraphCastPseudoHuberLossSchema(BaseLossSchema):
    """Schema for GraphCast-style Pseudo-Huber loss (smooth L1)."""

    delta: float = 10.0
    "Transition point from quadratic to linear behavior."


class GraphCastCRPSLossSchema(BaseLossSchema):
    """Schema for GraphCast-style fair-CRPS loss (FGN-style ensemble training).

    Per-cell metric is fair/almost-fair kernel CRPS; the GraphCastBaseLoss
    reduction (mean-over-levels-per-group → sum-over-grid → mean-over-ensemble
    → sum-over-groups → weighted-batch-mean) is applied unchanged on top.
    """

    alpha: float = 1.0
    """Blend between fair (1.0; FGN) and unfair (0.0; MAE/N) CRPS. epsilon = (1 - alpha) / N."""
    no_autocast: bool = True
    "Deactivate autocast for the kernel CRPS calculation (matches AlmostFairKernelCRPS)."


class GraphCastGaussianNLLLossSchema(BaseLossSchema):
    """Schema for GraphCast-style Gaussian NLL loss with learned diagonal covariance."""

    variance_mode: Literal["per_variable", "per_variable_per_level"] = "per_variable"
    "How to parameterize variances: 'per_variable' or 'per_variable_per_level'."
    variance_init: float = 0.0
    "Initial value for log_var (var = exp(log_var)). Default 0.0 (var=1, equivalent to MSE)."
    variance_eps: float = 1e-6
    "Small constant added to variance for numerical stability."
    variance_trainable: bool = True
    "Whether to learn the variances. If False, keeps them fixed at init."
    variance_min: float | None = None
    "Minimum allowed variance (prevents collapse). If None, no minimum constraint. Recommended: 1e-4."
    variance_max: float | None = None
    "Maximum allowed variance (prevents explosion). If None, no maximum constraint. Recommended: 100.0."
    sample_weighting: bool = False
    "If True, downweight samples with extreme target values."
    sample_weight_threshold: float = 10.0
    "Target magnitude threshold for weighting."
    sample_weight_min: float = 0.01
    "Minimum weight for extreme samples."


class CombinedLossSchema(BaseLossSchema):
    # Accept any concrete LossSchema (spatial, graphcast, base, etc.).
    # list[BaseLossSchema] alone only matched items with an enum target_,
    # which excluded LogFFT2Distance / FourierCorrelationLoss.
    losses: list[
        BaseLossSchema
        | HuberLossSchema
        | GraphCastMSELossSchema
        | GraphCastHuberLossSchema
        | GraphCastLogCoshLossSchema
        | GraphCastClippedMSELossSchema
        | GraphCastPseudoHuberLossSchema
        | GraphCastGaussianNLLLossSchema
        | LogFFT2DistanceSchema
        | FourierCorrelationLossSchema
        | SpectralAmplitudeLossSchema
        | SpatialGradientLossSchema
        | SpectralLossSchema
    ] = Field(min_length=1)
    "Losses to combine, can be any of the normal losses or a spatial/spectral loss."
    loss_weights: list[int | float] | None = None
    "Weightings of losses, if not set, all losses are weighted equally."

    @field_validator("losses", mode="before")
    @classmethod
    def add_empty_scalers(cls, losses: Any) -> Any:
        """Add empty scalers to loss functions, as scalers can be set at top level."""
        from omegaconf.omegaconf import open_dict

        for loss in losses:
            if "scalers" not in loss:
                with open_dict(loss):
                    loss["scalers"] = []
        return losses

    @model_validator(mode="after")
    def check_length_of_weights_and_losses(self) -> Self:
        """Check that the number of losses and weights match, or if not set, skip."""
        losses, loss_weights = self.losses, self.loss_weights
        if loss_weights is not None and len(losses) != len(loss_weights):
            error_msg = "Number of losses and weights must match"
            raise ValueError(error_msg)
        return self


class GraphCastCombinedLossSchema(CombinedLossSchema):
    """GraphCastCombinedLoss is a CombinedLoss subclass that routes set_data_indices to children.

    Schema-wise it's identical to CombinedLossSchema — same losses list and
    loss_weights — it only differs at runtime.
    """

    target_: Literal[
        "anemoi.training.losses.graphcast_combined.GraphCastCombinedLoss",
    ] = Field(..., alias="_target_")


class _WrappedInnerLossSchema(BaseModel):
    """Minimal schema for an `inner_loss` dict inside a loss-wrapper (HGL, Wind).

    Accepts a DictConfig with a `_target_` key pointing to any known loss and
    any extra fields that the target loss class may require. Strict key
    checking is delegated to the target loss's own schema at runtime; the
    wrapper itself only enforces the presence of `_target_`.
    """

    model_config = {"extra": "allow", "populate_by_name": True}
    target_: str = Field(..., alias="_target_")


class HorizontalGradientLossSchema(BaseLossSchema):
    """Schema for HorizontalGradientLoss (FastNet Sec. 5 gradient augmentation)."""

    target_: Literal[
        "anemoi.training.losses.horizontal_gradient.HorizontalGradientLoss",
    ] = Field(..., alias="_target_")
    inner_loss: _WrappedInnerLossSchema = Field(...)
    "Inner loss to which gradient-augmented inputs are passed."
    x_dim: int = Field(..., example=246)
    y_dim: int = Field(..., example=246)
    n_vars: int = Field(..., example=117)
    "Number of output variables — sizes the online σ_∂x / σ_∂y buffers."
    raw_weight: float = 1.0
    dx_weight: float = 1.0
    dy_weight: float = 1.0
    normalize_gradients: bool = True
    distributed_stats: bool = True


class GraphCastWindAwareLossSchema(BaseLossSchema):
    """Schema for GraphCastWindAwareLoss (FastNet wind decomposition)."""

    target_: Literal[
        "anemoi.training.losses.graphcast_wind.GraphCastWindAwareLoss",
    ] = Field(..., alias="_target_")
    inner_loss: _WrappedInnerLossSchema = Field(...)
    "Inner loss applied to direction-decomposed + speed-only inputs."
    speed_weight: float = 5.0
    "FastNet Eq. 10: λ_speed multiplier on the speed contribution."
    epsilon: float = 1e-6
    "Small constant inside sqrt(u² + v² + ε²) to avoid s → 0 blow-ups."
    u_v_pairs: Optional[list[list[str]]] = None
    "Optional override of u/v variable-name pairs. If None, auto-detect."


class GraphCastFullLossSchema(BaseLossSchema):
    """Schema for GraphCastFullLoss — flat FastNet stack in one class.

    Replaces the HorizontalGradient → Wind → Combined[MSE+MSH] nested stack.
    Ablate any term by setting its weight to 0; zero-weight terms are not
    computed and (in the MSH/Welford cases) not instantiated.
    """

    target_: Literal[
        "anemoi.training.losses.graphcast_full.GraphCastFullLoss",
    ] = Field(..., alias="_target_")
    x_dim: int = Field(..., example=246)
    y_dim: int = Field(..., example=246)
    n_vars: int = Field(..., example=117)
    raw_mse_weight: float = 1.0
    raw_msh_weight: float = 1.0
    grad_x_weight: float = 1.0
    grad_y_weight: float = 1.0
    wind_speed_weight: float = 5.0
    wind_dir_weight: float = 1.0
    coherence_weight: float = 1.0
    use_gamma_k: bool = True
    gamma_k_min: float = 1.0
    use_variable_normalization: bool = True
    normalize_gradients: bool = True
    epsilon: float = 1.0e-6
    grad_var_weights: Optional[dict[str, NonNegativeFloat]] = None
    "Per-variable weights for the ∂x/∂y terms only. Keys are output-variable names "
    "(e.g. ``qv_33``) or level-stripped group stems (e.g. ``pressure``). 0 fully ablates "
    "that variable's gradient contribution; ``default`` (if set) covers unmatched names."
    u_v_pairs: Optional[list[list[str]]] = None
    distributed_stats: bool = True
    mse_scalers: Optional[list[str]] = None
    msh_scalers: Optional[list[str]] = None
    precomputed_stats_path: Optional[str] = None
    column_mass_flux_weight: NonNegativeFloat = 0.0
    "Weight on the column-mass-flux conservation term. Penalises domain-mean "
    "(Σ w_lev · Δp_lev) mismatch between prediction and target. 0 disables. "
    "Targets the compensating-subsidence failure mode: ML emulators often fail to "
    "produce mass-balanced compensating flow around convective columns, leading to "
    "spurious net upward mass flux that drives upper-level theta drift over rollout."
    w_var_names: Optional[list[str]] = None
    "Output-variable names representing W at each retained vertical level, ordered "
    "low-to-high in altitude. If None, autodetected by matching ``w_<level_int>``."
    w_level_pressure_weights: Optional[list[float]] = None
    "Δp weights (Pa) per W level for the column integral, same length and order as "
    "``w_var_names``. If None, uniform weighting is used (less physical but works). "
    "Typically computed from US Standard Atmosphere at each level's zeta height."
    hydrostatic_weight: NonNegativeFloat = 0.0
    "Weight on the hydrostatic-balance soft constraint. Penalises per-pixel deviation "
    "from the hypsometric equation at each adjacent level pair via an error-tolerant "
    "loss f(r/α) = (r/α)² / (1 + exp(1 - (r/α)²)). Below α the loss is ≈0 (tolerates "
    "GRAF's natural non-hydrostatic imbalance from convection); above α it approaches "
    "MSE. 0 disables. Targets the same lid-drift failure mode as column_mass_flux but "
    "by enforcing the underlying invariant directly."
    hydrostatic_alphas: Optional[list[float]] = None
    "Per-level-pair tolerance α_k (K) for the error-tolerant loss. Length must be "
    "len(hydrostatic_z_levels) - 1. Calibrated from the training-zarr natural "
    "distribution of r_k via α_k = Q_k(p) · √(W₀(1) + 1) ≈ Q_k(50%) × 1.252 "
    "(precompute_hydrostatic_alphas.py)."
    hydrostatic_z_levels: Optional[list[float]] = None
    "Fixed zeta-level heights (m), ordered low-to-high in altitude, one per resolved "
    "pressure/theta/qv output variable. dz_k = z_k - z_{k-1} is computed internally."
    hydrostatic_p_var_names: Optional[list[str]] = None
    "Output-variable names for pressure at each retained vertical level, ordered to "
    "match ``hydrostatic_z_levels``. If None, autodetected by matching "
    "``pressure_<level_int>`` and sorting by level."
    hydrostatic_theta_var_names: Optional[list[str]] = None
    "Output-variable names for potential temperature at each level, same ordering as "
    "``hydrostatic_p_var_names``. If None, autodetected by matching ``theta_<level_int>``."
    hydrostatic_qv_var_names: Optional[list[str]] = None
    "Output-variable names for water-vapor mixing ratio at each level, same ordering "
    "as ``hydrostatic_p_var_names``. If None, autodetected by matching ``qv_<level_int>``."
    "Path to anemoi zarr containing statistics_msh_beta / statistics_gradient_{x,y}_stdev. Skips online Welford."


LossSchemas = (
    BaseLossSchema
    | HuberLossSchema
    | CombinedLossSchema
    | GraphCastCombinedLossSchema
    | GraphCastFullLossSchema
    | HorizontalGradientLossSchema
    | GraphCastWindAwareLossSchema
    | AlmostFairKernelCRPSSchema
    | KernelCRPSSchema
    | SpectralLossSchema
    | MultiScaleLossSchema
    | GraphCastMSELossSchema
    | GraphCastHuberLossSchema
    | GraphCastLogCoshLossSchema
    | GraphCastClippedMSELossSchema
    | GraphCastPseudoHuberLossSchema
    | GraphCastGaussianNLLLossSchema
    | GraphCastCRPSLossSchema
    | LogFFT2DistanceSchema
    | FourierCorrelationLossSchema
    | SpectralAmplitudeLossSchema
    | SpatialGradientLossSchema
)


class ImplementedStrategiesUsingBaseDDPStrategySchema(str, Enum):
    ddp_ens = "anemoi.training.distributed.strategy.DDPEnsGroupStrategy"
    ddp = "anemoi.training.distributed.strategy.DDPGroupStrategy"


class BaseDDPStrategySchema(BaseModel):
    """Strategy configuration."""

    target_: ImplementedStrategiesUsingBaseDDPStrategySchema = Field(..., alias="_target_")
    num_gpus_per_model: PositiveInt = Field(example=2)
    "Number of GPUs per model."
    read_group_size: PositiveInt = Field(example=1)
    "Number of GPUs per reader group. Defaults to number of GPUs."
    
    # Added to profile whether they improve GPU sync
    gradient_as_bucket_view: bool = False
    static_graph: bool = False
    find_unused_parameters: bool = True


class DDPEnsGroupStrategyStrategySchema(BaseDDPStrategySchema):
    """Strategy object from anemoi.training.strategy."""

    num_gpus_per_ensemble: PositiveInt = Field(example=2)
    "Number of GPUs per ensemble."


StrategySchemas = BaseDDPStrategySchema | DDPEnsGroupStrategyStrategySchema


class BaseTrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = Field(example=None)
    "Run ID: used to resume a run from a checkpoint, either last.ckpt or specified in system.input.warm_start."
    fork_run_id: str | None = Field(example=None)
    "Run ID to fork from, either last.ckpt or specified in system.input.warm_start."
    load_weights_only: bool = Field(example=False)
    "Load only the weights from the checkpoint, not the optimiser state."
    transfer_learning: bool = Field(example=False)
    "Flag to activate transfer learning mode when loading a checkpoint."
    submodules_to_freeze: list[str] = Field(example=["processor"])
    "List of submodules to freeze during transfer learning."
    deterministic: bool = Field(default=False)
    "This flag sets the torch.backends.cudnn.deterministic flag. Might be slower, but ensures reproducibility."
    precision: str = Field(default="16-mixed")
    "Precision"
    multistep_input: PositiveInt = Field(example=2)
    """Number of input steps for the model. E.g. 1 = single step scheme, X(t-1) used to predict X(t),
    k > 1: multistep scheme, uses [X(t-k), X(t-k+1), ... X(t-1)] to predict X(t)."""
    accum_grad_batches: PositiveInt = Field(default=1)
    """Accumulates gradients over k batches before stepping the optimizer.
    K >= 1 (if K == 1 then no accumulation). The effective bacthsize becomes num-device * k."""
    num_sanity_val_steps: NonNegativeInt = Field(example=6)
    "Sanity check runs n batches of val before starting the training routine."
    gradient_clip: GradientClip
    "Config for gradient clipping."
    strategy: StrategySchemas
    "Strategy to use."
    swa: SWA = Field(default_factory=SWA)
    "Config for stochastic weight averaging."
    ewa: EWA = Field(default_factory=EWA)
    "Config for exponential weight averaging."
    input_noise_sigma: NonNegativeFloat = Field(default=0.0)
    """Standard deviation of Gaussian noise added to model inputs during training
    (in normalized state space). Only honored when ``training.model_task`` resolves
    to a forecaster that reads it (e.g. grafai.training.NoisyResidualForecaster).
    0.0 disables noise. Typical values: 0.01-0.10."""
    training_loss: LossSchemas
    "Training loss configuration."
    loss_gradient_scaling: bool = False
    "Dynamic rescaling of the loss gradient. Not yet tested."
    scalers: dict[str, ScalerSchema]
    "Scalers to use in the computation of the loss and validation scores."
    validation_metrics: dict[str, LossSchemas]
    "List of validation metrics configurations."
    variable_groups: dict[str, str | list[str] | dict[str, str | bool | list[str | int]]]
    "Groups for variable loss scaling"
    max_epochs: PositiveInt | None = None
    "Maximum number of epochs, stops earlier if max_steps is reached first."
    max_steps: PositiveInt = 150000
    "Maximum number of steps, stops earlier if max_epochs is reached first."
    lr: LR = Field(default_factory=LR)
    "Learning rate configuration."
    optimizer: OptimizerSchema = Field(default_factory=OptimizerSchema)
    "Optimizer configuration."
    recompile_limit: PositiveInt = 32
    "How many times torch.compile will recompile a function for a given input shape."
    metrics: list[str]
    "List of metrics"
    ensemble_size_per_device: PositiveInt = 1
    "Number of ensemble members per device. Default is 1 for non-ensemble forecasting."


class ForecasterSchema(BaseTrainingSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphForecaster", 
                        # Monte 7 Nov 2025: Added to accomodate 
                        # the GraphResidualForecaster.
                        "anemoi.training.train.tasks.GraphResidualForecaster"
                       ] = Field(..., alias="model_task")
    "Training objective."
    rollout: Rollout = Field(default_factory=Rollout)
    "Rollout configuration."

class ForecasterEnsSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphEnsForecaster",] = Field(..., alias="model_task")
    "Training objective."


class EnsResidualForecasterSchema(ForecasterSchema):
    # FGN-style ensemble residual forecaster: combines GraphResidualForecaster's
    # residual reconstruction with GraphEnsForecaster's ensemble path + per-member
    # noise injection through AnemoiDiTModel.forward_with_noise.
    model_task: Literal["anemoi.training.train.tasks.GraphEnsResidualForecaster"] = Field(
        ...,
        alias="model_task",
    )
    "Training objective (ensemble + residual + per-member noise; pair with GraphCastCRPSLoss)."
    noise_vector_dim: PositiveInt = Field(default=32)
    "Per-(batch, member) noise-vector dimensionality. Must match model.dit.noise_vector_dim. Default 32 (FGN)."


class DiffusionForecasterSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphDiffusionForecaster"] = Field(..., alias="model_task")
    "Training objective."


class DiffusionTendForecasterSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphDiffusionTendForecaster"] = Field(
        ...,
        alias="model_task",
    )
    "Training objective."


class InterpolationSchema(BaseTrainingSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphInterpolator"] = Field(..., alias="model_task")
    "Training objective."
    explicit_times: ExplicitTimes
    "Time indices for input and output."
    target_forcing: TargetForcing
    "Forcing parameters for target output times."


TrainingSchema = Annotated[
    ForecasterSchema
    | ForecasterEnsSchema
    | EnsResidualForecasterSchema
    | InterpolationSchema
    | DiffusionForecasterSchema
    | DiffusionTendForecasterSchema,
    Discriminator("model_task"),
]
