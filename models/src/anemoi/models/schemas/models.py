# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel

from .decoder import GNNDecoderSchema  # noqa: TC001
from .decoder import GraphInteractionNetDecoderSchema  # noqa: TC001
from .decoder import GraphTransformerDecoderSchema  # noqa: TC001
from .decoder import TransformerDecoderSchema  # noqa: TC001
from .encoder import GNNEncoderSchema  # noqa: TC001
from .encoder import GraphInteractionNetEncoderSchema  # noqa: TC001
from .encoder import GraphTransformerEncoderSchema  # noqa: TC001
from .encoder import TransformerEncoderSchema  # noqa: TC001
from .processor import BandedTransformerProcessorSchema  # noqa: TC001
from .processor import GNNProcessorSchema  # noqa: TC001
from .processor import GraphInteractionNetProcessorSchema  # noqa: TC001
from .processor import GraphTransformerProcessorSchema  # noqa: TC001
from .processor import PointWiseMLPProcessorSchema  # noqa: TC001
from .processor import TransformerProcessorSchema  # noqa: TC001
from .residual import ResidualConnectionSchema

LOGGER = logging.getLogger(__name__)


class DefinedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"
    ANEMOI_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC = "anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec"
    ANEMOI_ENS_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiEnsModelEncProcDec"
    ANEMOI_MODEL_ENC_HIERPROC_DEC = "anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical"
    ANEMOI_MODEL_ENC_HIERPROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDecHierarchical"
    ANEMOI_MODEL_INTERPENC_PROC_DEC = "anemoi.models.models.interpolator.AnemoiModelEncProcDecInterpolator"
    ANEMOI_MODEL_INTERPENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiModelEncProcDecInterpolator"
    ANEMOI_DIFFUSION_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.diffusion_encoder_processor_decoder.AnemoiDiffusionModelEncProcDec"
    )
    ANEMOI_DIFFUSION_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiDiffusionModelEncProcDec"
    ANEMOI_DIFFUSION_TEND_MODEL_ENC_PROC_DEC = (
        "anemoi.models.models.diffusion_encoder_processor_decoder.AnemoiDiffusionTendModelEncProcDec"
    )
    ANEMOI_DIFFUSION_TEND_MODEL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiDiffusionTendModelEncProcDec"
    
    ANEMOI_RESIDUAL_ENC_PROC_DEC = "anemoi.models.models.residual_encoder_processor_decoder.AnemoiResidualModelEncProcDec"
    ANEMOI_RESIDUAL_ENC_PROC_DEC_SHORT = "anemoi.models.models.AnemoiResidualModelEncProcDec"

    ANEMOI_DIT_MODEL = "anemoi.models.models.dit_wrapper.AnemoiDiTModel"
    ANEMOI_DIT_MODEL_SHORT = "anemoi.models.models.AnemoiDiTModel"

    ANEMOI_UNET_MODEL = "anemoi.models.models.unet_wrapper.AnemoiUNetModel"
    ANEMOI_UNET_MODEL_SHORT = "anemoi.models.models.AnemoiUNetModel"

    # v30 Atlas-style latent rollout architecture (Variant B normalization).
    ANEMOI_ATLAS_DECODER_MODEL = "anemoi.models.models.decoder_dit_wrapper.AnemoiDecoderDiTModel"
    ANEMOI_ATLAS_DECODER_MODEL_SHORT = "anemoi.models.models.AnemoiDecoderDiTModel"
    ANEMOI_ATLAS_LATENT_MODEL = "anemoi.models.models.latent_dit_wrapper.AnemoiLatentDiTModel"
    ANEMOI_ATLAS_LATENT_MODEL_SHORT = "anemoi.models.models.AnemoiLatentDiTModel"
    ANEMOI_ATLAS_COMPOSED_MODEL = "anemoi.models.models.atlas_composed_model.AnemoiAtlasModel"
    ANEMOI_ATLAS_COMPOSED_MODEL_SHORT = "anemoi.models.models.AnemoiAtlasModel"

class Model(BaseModel):
    target_: DefinedModels = Field(..., alias="_target_")
    "Model object defined in anemoi.models.model."
    convert_: str = Field("all", alias="_convert_")
    "The target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."


class DiffusionModel(Model):
    diffusion: DiffusionSchema = Field(default=None)
    "Diffusion configuration for diffusion models"


class TrainableParameters(PydanticBaseModel):
    data: NonNegativeInt = Field(example=8)
    "Size of the learnable data node tensor. Default to 8."
    hidden: NonNegativeInt = Field(example=8)
    "Size of the learnable hidden node tensor. Default to 8."


class ReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.ReluBounding"] = Field(..., alias="_target_")
    "Relu bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the Relu method."


class LeakyReluBoundingSchema(ReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky Relu bounding object defined in anemoi.models.layers.bounding."


class FractionBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.FractionBounding"] = Field(..., alias="_target_")
    "Fraction bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh fraction method."
    min_val: float
    "The minimum value for the HardTanh activation. Correspond to the minimum fraction of the total_var."
    max_val: float
    "The maximum value for the HardTanh activation. Correspond to the maximum fraction of the total_var."
    total_var: str
    "Variable from which the secondary variables are derived. \
    For example, convective precipitation should be a fraction of total precipitation."


class LeakyFractionBoundingSchema(FractionBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyFractionBounding"] = Field(..., alias="_target_")
    "Leaky fraction bounding object defined in anemoi.models.layers.bounding."


class HardtanhBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.HardtanhBounding"] = Field(..., alias="_target_")
    "Hard tanh bounding method function from anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh method."
    min_val: float
    "The minimum value for the HardTanh activation."
    max_val: float
    "The maximum value for the HardTanh activation."


class LeakyHardtanhBoundingSchema(HardtanhBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyHardtanhBounding"] = Field(..., alias="_target_")
    "Leaky hard tanh bounding method function from anemoi.models.layers.bounding."


class NormalizedReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.NormalizedReluBounding"] = Field(..., alias="_target_")
    variables: list[str]
    min_val: list[float]
    normalizer: list[str]

    @model_validator(mode="after")
    def check_num_normalizers_and_min_val_matches_num_variables(self) -> NormalizedReluBoundingSchema:
        error_msg = f"""{self.__class__} requires that number of normalizers ({len(self.normalizer)}) or
        match the number of variables ({len(self.variables)})"""
        assert len(self.normalizer) == len(self.variables), error_msg
        error_msg = f"""{self.__class__} requires that number of min_val ({len(self.min_val)}) or  match
        the number of variables ({len(self.variables)})"""
        assert len(self.min_val) == len(self.variables), error_msg
        return self


class NormalizedLeakyReluBoundingSchema(NormalizedReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.NormalizedLeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky normalized Relu bounding object defined in anemoi.models.layers.bounding."


Bounding = Annotated[
    Union[
        ReluBoundingSchema,
        LeakyReluBoundingSchema,
        FractionBoundingSchema,
        LeakyFractionBoundingSchema,
        HardtanhBoundingSchema,
        LeakyHardtanhBoundingSchema,
        NormalizedReluBoundingSchema,
        NormalizedLeakyReluBoundingSchema,
    ],
    Field(discriminator="target_"),
]


class NoOutputMaskSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.NoOutputMask"] = Field(..., alias="_target_")


class Boolean1DSchema(BaseModel):
    target_: Literal["anemoi.training.utils.masks.Boolean1DMask"] = Field(..., alias="_target_")
    nodes_name: str = Field(examples="data")
    attribute_name: str = Field(example="cutout_mask")


OutputMaskSchemas = Union[NoOutputMaskSchema, Boolean1DSchema]


class DiffusionSchema(BaseModel):
    sigma_data: PositiveFloat = Field(default=1.0, examples=[1.0])
    "Data scaling parameter"
    noise_channels: PositiveInt = Field(default=32, examples=[32])
    "Number of channels for noise embedding"
    noise_cond_dim: PositiveInt = Field(default=16, examples=[16])
    "Dimension of noise conditioning"
    sigma_max: PositiveFloat = Field(default=100.0, examples=[100.0])
    "Maximum noise level for training"
    sigma_min: PositiveFloat = Field(default=0.02, examples=[0.02])
    "Minimum noise level for training"
    rho: PositiveFloat = Field(default=7.0, examples=[7.0])
    "Karras schedule parameter for training noise distribution"
    noise_embedder: dict = Field(default_factory=dict)
    "Noise embedder configuration with _target_ for Hydra instantiation"
    inference_defaults: dict = Field(default_factory=dict)
    "Default parameters for inference sampling"


class DiTConfigSchema(BaseModel):
    """Schema for DiT-specific configuration."""

    field_shape: list[PositiveInt]
    "Spatial dimensions of the data grid [H, W]."
    mode: Literal["deterministic", "probabilistic"] = Field(default="deterministic")
    "Training mode: deterministic (MSE) or probabilistic (diffusion)."
    patch_size: PositiveInt = Field(default=4, examples=[4, 6])
    "Patch size for tokenization. Each token covers patch_size x patch_size grid cells."
    hidden_size: PositiveInt = Field(default=512, examples=[384, 512, 768])
    "Transformer embedding dimension."
    depth: PositiveInt = Field(default=16, examples=[12, 16, 24])
    "Number of DiT transformer blocks."
    num_heads: PositiveInt = Field(default=8, examples=[6, 8, 12])
    "Number of attention heads."
    mlp_ratio: float = Field(default=4.0)
    "MLP hidden dimension multiplier."
    attention_backend: Literal["timm", "natten2d", "transformer_engine"] = Field(default="natten2d")
    "Attention backend. natten2d for O(N) neighborhood attention."
    conditioning_embedder: Literal["zero", "dit", "edm"] = Field(default="dit")
    "Conditioning type. dit=late-fusion (default), edm=early-fusion, zero=bias-only (not compatible with ProjLayer detokenizer)."
    condition_dim: Optional[int] = Field(default=None)
    "External conditioning dimension. None for unconditional."
    tokenizer_kwargs: dict = Field(default_factory=dict)
    "Kwargs passed to PatchEmbed2DTokenizer (e.g., pos_embed='none')."
    tokenizer_kernel_size: Optional[int] = Field(default=None)
    "Tokenizer Conv2d kernel size in input cells. None (default) uses kernel=patch_size for the standard non-overlapping PatchEmbed (each token is a disjoint patch). Set to a larger even-offset value (e.g. 8 with patch_size=4) to enable an OverlappingPatchEmbed2DTokenizer where adjacent tokens share input cells, structurally breaking the per-token spatial independence that drives 16 km decoder pixelation. kernel_size must be >= patch_size and (kernel_size − patch_size) must be even."
    attn_kwargs: dict = Field(default_factory=dict)
    "Kwargs passed to attention module (e.g., attn_kernel=13)."
    conditioning_embedder_kwargs: dict = Field(default_factory=dict)
    "Kwargs passed to conditioning embedder."
    force_tokenization_fp32: bool = Field(default=True)
    "Force tokenizer/detokenizer to run in fp32 for numerical stability."
    # Conv refinement after detokenizer (smooths patch-boundary artifacts)
    conv_refinement_blocks: int = Field(default=0)
    "Number of conv3x3-GELU-conv3x3 refinement blocks after DiT detokenizer (0=disabled). Zero-init final layer so it starts as identity residual."
    conv_refinement_kernel: int = Field(default=3)
    "Kernel size for conv refinement blocks."
    conv_refinement_hidden: int = Field(default=0)
    "Hidden channel width in each refinement block (0=use num_output_channels)."
    activation: str = Field(default="gelu")
    "Activation for the DiT transformer blocks. One of 'gelu' (default, existing checkpoints), 'silu', 'relu', 'leaky_relu'. Implemented as a post-init nn.GELU -> nn.<activation> swap inside the physicsnemo DiT."
    conv_refinement_activation: Optional[str] = Field(default=None)
    "Activation inside the conv_refinement block. If None (default), follows `activation`. One of 'gelu', 'silu', 'relu', 'leaky_relu'."
    conv_refinement_init: Literal["default", "gaussian", "gaussian_lowpass"] = Field(default="default")
    "Init of the first Conv2d in each refinement block. 'default'=Kaiming; 'gaussian' / 'gaussian_lowpass' seed with a normalised 2-D Gaussian kernel (σ = conv_refinement_init_sigma) so the refinement starts as a smoothing filter rather than random."
    conv_refinement_init_sigma: PositiveFloat = Field(default=0.7)
    "σ (in pixels) for the gaussian / gaussian_lowpass init. Ignored when conv_refinement_init='default'."
    # Fixed depth-wise Gaussian LPF right after the DiT detokenizer (anti-aliasing, non-learnable)
    detokenizer_lowpass_sigma: float = Field(default=0.0)
    "σ of the post-detokenizer Gaussian low-pass filter. 0 (default) disables; positive float (typical 0.5–1.0) enables a non-learnable, depth-wise 2-D Gaussian blur as a Nyquist rolloff. Applied in-line before conv_refinement."
    detokenizer_lowpass_kernel: int = Field(default=5)
    "Odd kernel size (3/5/7) for the Gaussian LPF. Used only when detokenizer_lowpass_sigma > 0."
    detokenizer_type: Literal[
        "linear_reshape",
        "pixel_shuffle",
        "pixel_shuffle_3x3x2",
        "pixel_shuffle_5x5x2",
        "pixel_shuffle_7x7x1",
        "conv_transpose_k12_s4",
        "bilinear_3x3x2",
        "hierarchical_2stage",
    ] = Field(default="linear_reshape")
    "Detokenizer head architecture. 'linear_reshape' (default) is the stock DiT ProjLayer + reshape (per-token Linear, no cross-patch blending; structurally pixelates). The other variants are Tier-A1/A3 cross-patch-mixing heads from the literature survey: 'pixel_shuffle' (alias for 3x3x2), 'pixel_shuffle_5x5x2' (RF=9 cells), 'pixel_shuffle_7x7x1' (single 7x7 conv, RF=7), 'conv_transpose_k12_s4' (3x overlap, no Odena checkerboard), 'bilinear_3x3x2' (smooth + learned recovery), 'hierarchical_2stage' (SegFormer-style two-step PixelShuffle)."
    # Diffusion-specific (only used when mode='probabilistic')
    sigma_data: Optional[PositiveFloat] = Field(default=1.0)
    "Data scaling parameter for EDM preconditioning."
    sigma_max: Optional[PositiveFloat] = Field(default=100.0)
    "Maximum noise level for diffusion training."
    sigma_min: Optional[PositiveFloat] = Field(default=0.02)
    "Minimum noise level for diffusion training."
    rho: Optional[PositiveFloat] = Field(default=7.0)
    "Karras schedule parameter for training noise distribution."
    inference_defaults: dict = Field(default_factory=dict)
    "Default parameters for inference sampling (noise schedule, sampler)."
    output_mode: Literal["residual", "state"] = Field(default="residual")
    "How the model output is interpreted. 'residual' (default, back-compat): " \
    "DiT output is a normalised residual; the task reconstructs state externally. " \
    "'state': DiT output + input-state skip = predicted state in normalised space; " \
    "use with default GraphForecaster task. Boundings are applied only in 'state' mode."
    attn_drop_rate: float = Field(default=0.0)
    "Attention-weights dropout rate (physicsnemo DiTBlock). 0.0 = off (default). Used for U-Cast MC-dropout CRPS recipe (Cachay et al, arXiv 2604.09041)."
    proj_drop_rate: float = Field(default=0.0)
    "Post-attention projection dropout rate. 0.0 = off (default)."
    drop_path_rate: float = Field(default=0.0)
    "Stochastic-depth (DropPath) rate applied at each block's residual gate. 0.0 = off (default)."
    noise_vector_dim: Optional[int] = Field(default=None)
    "Dimension of the per-(batch, member) noise vector for FGN-style ensemble training (None = disabled, deterministic). 32 matches FGN."
    noise_encoder_type: Literal["matmul", "fourier_mlp", "none"] = Field(default="none")
    "How to encode the noise vector into the DiT hidden_size: 'matmul' (single Linear, FGN-faithful), 'fourier_mlp' (Sinusoidal+MLP, GenCast-style), 'none' (disabled). When 'matmul' or 'fourier_mlp', AnemoiDiTModel swaps the conditioning_embedder for a passthrough so the encoded noise reaches every adaLN unchanged."


class DiTModel(BaseModel):
    """Model target schema for DiT models."""

    target_: Literal[
        DefinedModels.ANEMOI_DIT_MODEL,
        DefinedModels.ANEMOI_DIT_MODEL_SHORT,
    ] = Field(..., alias="_target_")
    "DiT model object."
    convert_: str = Field("all", alias="_convert_")
    "The target's parameters to convert to primitive containers."
    dit: DiTConfigSchema
    "DiT-specific configuration."


class DiTModelSchema(PydanticBaseModel):
    """Schema for DiT models — replaces enc-proc-dec with a single DiT.

    Unlike BaseModelSchema, this does NOT require processor, encoder, or decoder
    fields because the DiT replaces the entire pipeline.
    """

    num_channels: NonNegativeInt = Field(example=512)
    "Feature tensor size (DiT hidden_size)."
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded."
    model: DiTModel = Field(...)
    "DiT model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters (typically all 0 for DiT)."
    bounding: list[Bounding]
    "List of bounding configurations applied to specified variables."
    output_mask: OutputMaskSchemas
    "Output mask configuration."
    residual: ResidualConnectionSchema = Field(..., discriminator="target_")
    "Residual connection schema."
    attributes: Optional[dict] = Field(default_factory=dict)
    "Node/edge attributes (typically empty for DiT)."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to be compiled."


class UNetConfigSchema(BaseModel):
    """Schema for SongUNet configuration."""

    field_shape: list[PositiveInt]
    "Spatial dimensions of the data grid [H, W]."
    mode: Literal["deterministic", "probabilistic"] = Field(default="deterministic")
    "Training mode: deterministic (MSE/MAE) or probabilistic (diffusion)."
    model_channels: PositiveInt = Field(default=128, examples=[128, 192, 320])
    "Base channel width. Channels at level i = model_channels * channel_mult[i]."
    channel_mult: list[PositiveInt] = Field(default=[1, 2, 3, 4])
    "Channel multipliers per level. Length determines number of U-Net levels."
    num_blocks: PositiveInt = Field(default=4, examples=[2, 4])
    "Number of residual convolutional blocks per level."
    n_attn_levels: PositiveInt = Field(default=2)
    "Number of coarsest resolution levels to apply self-attention."
    dropout: float = Field(default=0.10)
    "Dropout probability in U-Net blocks."
    encoder_type: Literal["standard", "skip", "residual"] = Field(default="standard")
    "Encoder architecture: standard (DDPM++), residual (NCSN++), skip."
    decoder_type: Literal["standard", "skip"] = Field(default="standard")
    "Decoder architecture: standard or skip."
    bottleneck_attention: bool = Field(default=True)
    "Apply self-attention at the bottleneck (innermost level)."
    large_kernel_stem: int = Field(default=0)
    "If >0, replace SongUNet's 3x3 stem with a depthwise-separable Kx K conv (RepLKNet-style); 0=disabled, 51=recommended."
    domain_parallel_size: PositiveInt = Field(default=1)
    "Number of GPUs for domain-parallel sharding (1=disabled). Splits spatial dimension across GPUs for large domains."
    shard_dim: PositiveInt = Field(default=2)
    "Spatial dimension to shard for domain parallelism (2=height, 3=width)."
    # Diffusion-specific (probabilistic mode)
    sigma_data: Optional[PositiveFloat] = Field(default=1.0)
    "Data scaling parameter for EDM preconditioning."
    sigma_max: Optional[PositiveFloat] = Field(default=100.0)
    "Maximum noise level for diffusion training."
    sigma_min: Optional[PositiveFloat] = Field(default=0.02)
    "Minimum noise level for diffusion training."
    rho: Optional[PositiveFloat] = Field(default=7.0)
    "Karras schedule parameter."
    inference_defaults: dict = Field(default_factory=dict)
    "Default parameters for inference sampling."


class UNetModel(BaseModel):
    """Model target schema for UNet models."""

    target_: Literal[
        DefinedModels.ANEMOI_UNET_MODEL,
        DefinedModels.ANEMOI_UNET_MODEL_SHORT,
    ] = Field(..., alias="_target_")
    "UNet model object."
    convert_: str = Field("all", alias="_convert_")
    "The target's parameters to convert to primitive containers."
    unet: UNetConfigSchema
    "UNet-specific configuration."


class UNetModelSchema(PydanticBaseModel):
    """Schema for UNet models — replaces enc-proc-dec with a single SongUNet."""

    num_channels: NonNegativeInt = Field(example=128)
    "Base channel width (UNet model_channels)."
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded."
    model: UNetModel = Field(...)
    "UNet model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters (typically all 0 for UNet)."
    bounding: list[Bounding]
    "List of bounding configurations applied to specified variables."
    output_mask: OutputMaskSchemas
    "Output mask configuration."
    residual: ResidualConnectionSchema = Field(..., discriminator="target_")
    "Residual connection schema."
    attributes: Optional[dict] = Field(default_factory=dict)
    "Node/edge attributes (typically empty for UNet)."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to be compiled."


class BaseModelSchema(PydanticBaseModel):
    num_channels: NonNegativeInt = Field(example=512)
    "Feature tensor size in the hidden space."
    keep_batch_sharded: bool = Field(default=True)
    "Keep the input batch and the output of the model sharded"
    model: Model = Field(default_factory=Model)
    "Model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters."
    bounding: list[Bounding]
    "List of bounding configuration applied in order to the specified variables."
    output_mask: OutputMaskSchemas  # !TODO CHECK!
    "Output mask"
    latent_skip: bool = True
    "Add skip connection in latent space before/after processor. Currently only in interpolator."
    processor: Union[
        GNNProcessorSchema, GraphInteractionNetProcessorSchema, GraphTransformerProcessorSchema, TransformerProcessorSchema, PointWiseMLPProcessorSchema, BandedTransformerProcessorSchema
    ] = Field(
        ...,
        discriminator="target_",
    )
    "GNN processor schema."
    encoder: Union[GNNEncoderSchema, GraphInteractionNetEncoderSchema, GraphTransformerEncoderSchema, TransformerEncoderSchema] = Field(
        ...,
        discriminator="target_",
    )
    "GNN encoder schema."
    decoder: Union[GNNDecoderSchema, GraphInteractionNetDecoderSchema, GraphTransformerDecoderSchema, TransformerDecoderSchema] = Field(
        ...,
        discriminator="target_",
    )
    "GNN decoder schema.",
    residual: ResidualConnectionSchema = Field(
        ...,
        discriminator="target_",
    )
    "Residual connection schema."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to be compiled"


class NoOpNoiseInjectorSchema(BaseModel):
    """Schema for NoOpNoiseInjector - passes input through unchanged."""

    target_: Literal["anemoi.models.layers.ensemble.NoOpNoiseInjector"] = Field(..., alias="_target_")
    "No-op noise injector class"


class NoiseConditioningSchema(BaseModel):
    """Schema for NoiseConditioning - generates noise for conditioning."""

    target_: Literal["anemoi.models.layers.ensemble.NoiseConditioning"] = Field(..., alias="_target_")
    "Noise conditioning layer class"
    noise_std: NonNegativeInt = Field(example=1)
    "Standard deviation of the noise to be injected."
    noise_channels_dim: NonNegativeInt = Field(example=4)
    "Number of channels in the noise tensor."
    noise_mlp_hidden_dim: NonNegativeInt = Field(example=8)
    "Hidden dimension of the MLP used to process the noise."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"
    noise_matrix: Optional[str] = Field(default=None)
    "Path to the noise projection matrix file (.npz). If None, no projection is applied."
    transpose_noise_matrix: bool = Field(default=False)
    "Whether to transpose the noise projection matrix."
    row_normalize_noise_matrix: bool = Field(default=False)
    "Whether to row-normalize the noise projection matrix weights."
    autocast: bool = Field(default=False)
    "Whether to use autocast for the noise projection matrix operations."


class NoiseInjectorSchema(BaseModel):
    """Schema for NoiseInjector - injects noise directly into input tensor."""

    target_: Literal["anemoi.models.layers.ensemble.NoiseInjector"] = Field(..., alias="_target_")
    "Noise injector layer class"
    noise_std: NonNegativeInt = Field(example=1)
    "Standard deviation of the noise to be injected."
    noise_channels_dim: NonNegativeInt = Field(example=4)
    "Number of channels in the noise tensor."
    noise_mlp_hidden_dim: NonNegativeInt = Field(example=8)
    "Hidden dimension of the MLP used to process the noise."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


NoiseInjectorUnion = Annotated[
    Union[NoOpNoiseInjectorSchema, NoiseConditioningSchema, NoiseInjectorSchema],
    Field(discriminator="target_"),
]


class EnsModelSchema(BaseModelSchema):
    noise_injector: NoiseInjectorUnion = Field(...)
    "Noise injection configuration. Use NoOpNoiseInjector to disable, NoiseConditioning for conditioning, or NoiseInjector for direct injection."
    condition_on_residual: bool = Field(default=False)
    "Whether to condition the noise injection on the residual connection."


class DiffusionModelSchema(BaseModelSchema):
    model: DiffusionModel = Field(default_factory=DiffusionModel)
    "Diffusion Model schema"

    @model_validator(mode="after")
    def validate_no_bounding_for_diffusion(self) -> "DiffusionModelSchema":
        if self.bounding:
            msg = (
                "Diffusion models do not support bounding layers. "
                f"Found {len(self.bounding)} bounding configuration(s). "
                "Please remove all bounding configurations for diffusion models."
            )
            raise ValueError(msg)
        return self


class DiffusionTendModelSchema(DiffusionModelSchema):
    condition_on_residual: bool = Field(default=False)
    "Whether to condition the noise injection on the residual connection."


class HierarchicalModelSchema(BaseModelSchema):
    enable_hierarchical_level_processing: bool = Field(default=False)
    "Toggle to do message passing at every downscaling and upscaling step"
    level_process_num_layers: NonNegativeInt = Field(default=1)
    "Number of message passing steps at each level"


# ----------------------------------------------------------------------
# v30 Atlas-style architecture schemas (Variant B normalization).
# ----------------------------------------------------------------------


class AtlasDecoderConfigSchema(BaseModel):
    """Schema for the standalone Atlas decoder model.

    Maps to :class:`anemoi.models.models.decoder_dit_wrapper.AnemoiDecoderDiTModel`.
    """

    full_res_shape: list[PositiveInt] = Field(default=[250, 250])
    "Spatial extent of the full-resolution grid [H, W]."
    latent_shape: list[PositiveInt] = Field(default=[63, 63])
    "Spatial extent of the latent grid [h_lat, w_lat]. Must match the predictive model."
    in_channels_xt: PositiveInt
    "Channels in the full-resolution input x_t (prognostic + forcings)."
    in_channels_r: PositiveInt
    "Channels in the latent residual r_t (typically prognostic-only)."
    out_channels: PositiveInt
    "Channels in the predicted full-res residual delta_t (prognostic-only)."
    hidden_size: PositiveInt = Field(default=512)
    "DiT block hidden dimension."
    depth: PositiveInt = Field(default=8)
    "Number of DiT blocks."
    num_heads: PositiveInt = Field(default=8)
    "Attention heads. Must divide hidden_size."
    attn_kernel: PositiveInt = Field(default=9)
    "NATTEN local-attention kernel size (Atlas uses 3; we use 7–11 at 4 km)."
    embed_split: float = Field(default=0.5)
    "Fraction of hidden_size allocated to the x_t branch; rest goes to r_t."


class AtlasLatentConfigSchema(BaseModel):
    """Schema for the Atlas latent predictive model.

    Maps to :class:`anemoi.models.models.latent_dit_wrapper.AnemoiLatentDiTModel`.
    Variant B: emits latent residuals in mean-std space (NOT tendency-normalized).
    Tendency normalization happens in the loss via LatentVarTendencyScaler.
    """

    latent_shape: list[PositiveInt] = Field(default=[63, 63])
    "Spatial extent of the latent grid [h_lat, w_lat]."
    in_channels: PositiveInt
    "Prognostic channels per latent state."
    out_channels: PositiveInt
    "Channels in predicted latent residual (typically = in_channels, prognostic-only)."
    forcings_channels: NonNegativeInt = Field(default=0)
    "Forcing channels concatenated at the tokenizer input (0 disables; "
    "GRAF-AI uses 11 to inject HGT, land/sea, lat/lon, time-of-day, etc.)."
    hidden_size: PositiveInt = Field(default=512)
    "DiT block hidden dimension."
    depth: PositiveInt = Field(default=16)
    "Number of DiT blocks."
    num_heads: PositiveInt = Field(default=8)
    "Attention heads."
    history_len: Literal[1, 2] = Field(default=2)
    "Number of input history states (Atlas uses 2 = z_t, z_{t-1})."
    noise_vector_dim: NonNegativeInt = Field(default=32)
    "FGN noise-vector dim (32=FGN, 256+=Atlas, 0=deterministic)."


class AtlasComposedConfigSchema(BaseModel):
    """Schema for the composed Atlas model (encoder+predictive+decoder).

    Maps to :class:`anemoi.models.models.atlas_composed_model.AnemoiAtlasModel`.
    Inference-time wrapper that hosts the trained predictive + decoder.
    """

    full_res_shape: list[PositiveInt] = Field(default=[250, 250])
    "Full-res spatial extent."
    latent_shape: list[PositiveInt] = Field(default=[63, 63])
    "Latent spatial extent."
    prognostic_channels: PositiveInt
    "Number of prognostic channels (decoded by the decoder)."
    forcings_channels: NonNegativeInt = Field(default=0)
    "Number of forcing channels in x_t after the prognostic slice."


class AtlasDecoderModel(BaseModel):
    target_: Literal[
        DefinedModels.ANEMOI_ATLAS_DECODER_MODEL,
        DefinedModels.ANEMOI_ATLAS_DECODER_MODEL_SHORT,
    ] = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")
    decoder: AtlasDecoderConfigSchema = Field(...)


class AtlasLatentModel(BaseModel):
    target_: Literal[
        DefinedModels.ANEMOI_ATLAS_LATENT_MODEL,
        DefinedModels.ANEMOI_ATLAS_LATENT_MODEL_SHORT,
    ] = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")
    latent: AtlasLatentConfigSchema = Field(...)


class AtlasComposedModel(BaseModel):
    target_: Literal[
        DefinedModels.ANEMOI_ATLAS_COMPOSED_MODEL,
        DefinedModels.ANEMOI_ATLAS_COMPOSED_MODEL_SHORT,
    ] = Field(..., alias="_target_")
    convert_: str = Field("all", alias="_convert_")
    atlas: AtlasComposedConfigSchema = Field(...)


class AtlasDecoderModelSchema(PydanticBaseModel):
    """Top-level schema for the standalone Atlas decoder task.

    Mirrors DiTModelSchema (no enc-proc-dec pipeline) but with sensible
    defaults for the boundary / residual / output_mask fields that
    BaseGraphModule expects, since Variant B doesn't use them.
    """

    num_channels: NonNegativeInt = Field(default=512)
    "DiT hidden dimension."
    keep_batch_sharded: bool = Field(default=True)
    "Keep input batch + model output sharded across GPUs."
    model: AtlasDecoderModel = Field(...)
    "Decoder model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=lambda: TrainableParameters(data=0, hidden=0))
    "Unused for Atlas (no learned node attributes); default to zeros."
    bounding: list[Bounding] = Field(default_factory=list)
    "Empty by default; Variant B does no bounding (no ResidualNormalizer to clip against)."
    output_mask: OutputMaskSchemas
    "Output mask configuration."
    attributes: Optional[dict] = Field(default_factory=dict)
    "Unused for Atlas."
    compile: Optional[list[dict[str, Any]]] = Field(None)
    "Modules to compile."


class AtlasLatentModelSchema(PydanticBaseModel):
    """Top-level schema for the standalone Atlas latent predictive task."""

    num_channels: NonNegativeInt = Field(default=512)
    keep_batch_sharded: bool = Field(default=True)
    model: AtlasLatentModel = Field(...)
    trainable_parameters: TrainableParameters = Field(default_factory=lambda: TrainableParameters(data=0, hidden=0))
    bounding: list[Bounding] = Field(default_factory=list)
    output_mask: OutputMaskSchemas
    attributes: Optional[dict] = Field(default_factory=dict)
    compile: Optional[list[dict[str, Any]]] = Field(None)


class AtlasComposedModelSchema(PydanticBaseModel):
    """Top-level schema for the composed Atlas inference model.

    Used at inference time (and never at training, since the two submodels
    are trained separately). Holds metadata to assemble the composed model
    after loading the two checkpoints.
    """

    num_channels: NonNegativeInt = Field(default=512)
    keep_batch_sharded: bool = Field(default=True)
    model: AtlasComposedModel = Field(...)
    trainable_parameters: TrainableParameters = Field(default_factory=lambda: TrainableParameters(data=0, hidden=0))
    bounding: list[Bounding] = Field(default_factory=list)
    output_mask: OutputMaskSchemas
    attributes: Optional[dict] = Field(default_factory=dict)
    compile: Optional[list[dict[str, Any]]] = Field(None)


ModelSchema = Union[
    DiTModelSchema, UNetModelSchema, BaseModelSchema, EnsModelSchema, HierarchicalModelSchema, DiffusionModelSchema, DiffusionTendModelSchema,
    AtlasDecoderModelSchema, AtlasLatentModelSchema, AtlasComposedModelSchema,
]
