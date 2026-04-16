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
    attn_kwargs: dict = Field(default_factory=dict)
    "Kwargs passed to attention module (e.g., attn_kernel=13)."
    conditioning_embedder_kwargs: dict = Field(default_factory=dict)
    "Kwargs passed to conditioning embedder."
    force_tokenization_fp32: bool = Field(default=True)
    "Force tokenizer/detokenizer to run in fp32 for numerical stability."
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


ModelSchema = Union[
    DiTModelSchema, UNetModelSchema, BaseModelSchema, EnsModelSchema, HierarchicalModelSchema, DiffusionModelSchema, DiffusionTendModelSchema
]
