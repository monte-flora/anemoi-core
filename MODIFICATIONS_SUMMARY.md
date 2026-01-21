# Anemoi-Core Modifications Summary

## Overview
This document summarizes all modifications made to the anemoi-core repository for implementing GraphCast-style residual prediction for storm-scale limited-area weather forecasting.

---

## Modified Files (25 files)

### 1. **Core Model Architecture**

#### `models/src/anemoi/models/models/residual_encoder_processor_decoder.py`
**Purpose**: Remove skip connections to convert from skip-connection model to pure residual prediction

**Changes**:
- **Line 106-108**: Commented out skip connection in `_assemble_output()`
  - Removes: `x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]`
  - Residual now added in forecaster, not in model
- Added `_get_normalizer_buffers()` static method to extract norm_mul/norm_add from pre_processors
- Rewrote `predict()` method (lines 193-337) with full GraphCast-style residual workflow:
  - Step 1: Normalize input
  - Step 2: Model predicts normalized residuals
  - Step 3: Handle prognostic variables (residual prediction in physical space)
  - Step 4: Handle diagnostic variables (direct prediction)
  - Step 5: Gather for distributed processing
- Prognostic/diagnostic variable separation throughout prediction pipeline

---

### 2. **Training: Residual Forecaster**

#### `training/src/anemoi/training/train/tasks/residualforecaster.py`
**Purpose**: Implement GraphCast residual prediction with comprehensive diagnostics

**Major Changes**:
- **Lines 32-63**: Added diagnostic logging infrastructure
  - `_log_tensor_stats()`: Logs mean/std/min/max/abs_max for tensors
  - `_diag_log_interval = 100`: Log every 100 batches
  - `_diag_batch_counter`: Global counter
- **Lines 115-255**: Complete rewrite of `_rollout_step()`:
  - Separates prognostic and diagnostic variables throughout
  - Computes residuals in **physical space** (not normalized space)
  - `Δx_true_norm = (y_true_phys - x_last_phys) / σ_Δx`
  - Reconstructs predictions in physical space, then renormalizes
  - Comprehensive logging every 100 batches showing:
    - Normalized state tensors
    - Physical space tensors
    - Normalized residuals (target vs predicted)
    - Reconstruction quality
    - Prediction vs truth in normalized space
  - Loss computed in normalized residual space
  - Validation metrics computed in physical state space

**Key Insight**: This implementation matches WoFSCast's `InputsAndResiduals` wrapper behavior

---

### 3. **Model Layers: Decoder**

#### `models/src/anemoi/models/layers/mapper.py`
**Purpose**: Add optional final LayerNorm and debug zero-initialization

**Changes**:

**GraphTransformerBackwardMapper** (lines 666-759):
- Added `final_layer_norm: bool = True` parameter (line 669)
  - Lines 743-748: Conditional LayerNorm before final linear projection
  - If `final_layer_norm=False`: Direct linear projection (matches WoFSCast)
  - If `final_layer_norm=True`: LayerNorm → Linear (standard Anemoi)
- Enhanced zero-initialization logging (lines 749-759):
  - Logs when zero-initializing
  - Verifies weight/bias max values after initialization
- **Lines 116-125**: Debug logging in `BackwardMapperPostProcessMixin.post_process()`
  - Shows decoder input/output statistics
  - Helped diagnose zero-output bug

**GNNBackwardMapper** (lines 1036-1143):
- Added same parameters: `final_layer_norm`, `initialise_data_extractor_zero`
- Zero-initialization for GNN decoder (lines 1115-1123)
- Forward-time verification of zero weights (lines 1129-1139)

---

### 4. **Preprocessing: Residual Normalizer**

#### `models/src/anemoi/models/preprocessing/residual_normalizer.py`
**Purpose**: Normalize residuals using tendency statistics with float32 precision

**Changes**:
- **All computations now in float32** for numerical stability (lines 48-90)
- Fixed index bug: Uses `data.input.prognostic` (not `data.output.prognostic`) to index statistics
  - Line 97: `self._prog_idx = self.data_indices.data.input.prognostic`
- Added NaN/Inf debug infrastructure (lines 22-60, currently disabled)
- Methods updated for float32:
  - `transform()`: Compute normalized residual in float32, cast back to original dtype
  - `inverse_transform()`: Reconstruct physical state in float32, cast back
  - `transform_from_normalized()`: Work directly with normalized inputs

**Key Methods**:
- `transform(x_last_phys, y_true_phys)` → `Δx_norm`
- `inverse_transform(x_last_phys, Δx_norm)` → `y_pred_phys`

---

### 5. **Loss Functions**

#### `training/src/anemoi/training/losses/base.py`
**Purpose**: GraphCast variable grouping and optional sample weighting

**Changes**:
- Removed old debug code (lines 27-47 deleted)
- **Lines 315-360**: Added sample weighting to `GraphCastBaseLoss.__init__()`
  - `sample_weighting: bool = False`
  - `sample_weight_threshold: float = 10.0` (10-sigma threshold)
  - `sample_weight_min: float = 0.01` (1% minimum weight)
- **Lines 456-502**: Added `_compute_sample_weights()` method
  - Downweights samples with extreme target values
  - `weight = threshold / max(threshold, max_abs_target)`
  - Normalized to preserve loss scale
- Updated `_reduce()` to accept `sample_weights` parameter (line 537)
- Cleaned up reduction pipeline (removed debug checks)

**Reduction Order** (lines 510-528):
1. Mean over vertical levels within each variable group
2. Mean over ensemble and grid dimensions
3. Sum over variable groups
4. Weighted mean over batch (if sample_weights provided)

#### `training/src/anemoi/training/losses/graphcast_mse.py`
**Changes**: Updated to pass sample_weights through reduction chain (not shown in detail)

---

### 6. **Configuration Schemas**

#### `models/src/anemoi/models/schemas/decoder.py`
**Changes**:
- Added schema fields for both GNN and GraphTransformer decoders:
  - `initialise_data_extractor_zero: bool` (default: False)
  - `final_layer_norm: bool` (default: True)

#### `models/src/anemoi/models/schemas/models.py`
**Changes**:
- Added `BandedTransformerProcessorSchema` to processor union type (line 216)
  - Enables banded attention for processor (experimental)

#### `models/src/anemoi/models/schemas/processor.py`
**Changes**: Added BandedTransformerProcessorSchema (not detailed)

---

### 7. **Data Loading**

#### `training/src/anemoi/training/data/dataset.py`
**Purpose**: Implement trajectory-diverse batching for better training

**Changes**:
- **Line 33**: Added class variable `_trajectory_diverse_batching = True`
- **Lines 284-352**: Implemented `_get_trajectory_diverse_indices()` method
  - Reorders indices so consecutive samples come from different forecast trajectories
  - Ensures each batch contains diverse weather regimes
  - Algorithm:
    1. Group indices by trajectory ID
    2. Shuffle within each trajectory
    3. Round-robin across trajectories
  - Logs trajectory statistics per worker
- **Lines 370-373**: Applies trajectory-diverse reordering if enabled

**Motivation**: Prevents batches from being dominated by a single weather event

#### `training/src/anemoi/training/data/datamodule.py`
**Changes**:
- Lines 236-237: Commented out GPU configuration (debugging)
- Line 261: Added warning log when val_dataloader is called

---

### 8. **Graph Modifications**

#### `graphs/src/anemoi/graphs/edges/__init__.py`
- Added `MPASTopologicalEdges` import and export

#### `graphs/src/anemoi/graphs/nodes/__init__.py`
- Added `LimitedAreaMPASNodes` and `MPASCoarseNodes` imports

#### `graphs/src/anemoi/graphs/edges/builders/mpas.py` (NEW FILE)
- MPAS mesh topological edge builder

#### `graphs/src/anemoi/graphs/nodes/builders/from_mpas.py` (NEW FILE)
- MPAS mesh node builders for limited area and coarse meshes

**Purpose**: Support for MPAS (Model for Prediction Across Scales) mesh graphs

---

### 9. **Attention Mechanisms**

#### `models/src/anemoi/models/layers/attention.py`
**Changes**:
- **Lines 225-350** (estimated): Added `BandedGraphSelfAttention` class
  - Uses banded permutation with reverse Cuthill-McKee ordering
  - Makes graph adjacency local in sequence space
  - Enables windowed flash attention to approximate k-hop neighborhoods
- Added `compute_banded_permutation()` function (lines 225-250)

**Purpose**: Experimental optimization for large graphs with localized connectivity

#### `models/src/anemoi/models/layers/block.py`
**Changes**: Updated to support BandedGraphSelfAttention (not detailed)

#### `models/src/anemoi/models/layers/processor.py`
**Changes**: Updated for banded attention processor (not detailed)

---

### 10. **Other Files** (Minor changes)

#### `models/src/anemoi/models/preprocessing/imputer.py`
- Minor updates (not detailed in diff)

#### `models/src/anemoi/models/preprocessing/normalizer.py`
- Minor updates (not detailed in diff)

#### `training/src/anemoi/training/diagnostics/__init__.py`
- Minor updates (not detailed in diff)

#### `training/src/anemoi/training/diagnostics/callbacks/plot.py`
- Minor updates (not detailed in diff)

#### `training/src/anemoi/training/losses/__init__.py`
- Exports for new loss functions

#### `training/src/anemoi/training/schemas/training.py`
- Schema updates (not detailed)

#### `training/src/anemoi/training/train/train.py`
- Minor updates (not detailed)

#### `graphs/src/anemoi/graphs/export.py`
- MPAS export support

#### `graphs/src/anemoi/graphs/schemas/edge_schemas.py`
- MPAS edge schema

#### `graphs/src/anemoi/graphs/schemas/node_schemas.py`
- MPAS node schema

---

## New Files (10 files)

### 1. **Loss Functions**

#### `training/src/anemoi/training/losses/graphcast_huber.py` (186 lines)
**Purpose**: Huber loss with GraphCast reduction semantics

**Features**:
- Robust to outliers (transitions from quadratic to linear for large errors)
- Configurable delta threshold (recommended: 1.0 to 3.0)
- For |error| < δ: loss = 0.5 × error²
- For |error| ≥ δ: loss = δ × |error| - 0.5 × δ²
- Follows GraphCast variable grouping and reduction order
- Suitable for weather data with extreme events (100+ sigma outliers)

**Configuration**:
```yaml
training:
  training_loss:
    _target_: anemoi.training.losses.GraphCastHuberLoss
    delta: 3.0
    scalers: ['node_weights']
```

#### `training/src/anemoi/training/losses/graphcast_logcosh.py` (201 lines)
**Purpose**: Log-Cosh loss with GraphCast reduction

**Features**:
- Smooth approximation to Huber loss
- log(cosh(error)) ≈ 0.5 × error² for small errors
- log(cosh(error)) ≈ |error| - log(2) for large errors
- More smoothly differentiable than Huber

---

### 2. **MPAS Graph Support**

#### `graphs/src/anemoi/graphs/edges/builders/mpas.py`
- Edge builder for MPAS meshes

#### `graphs/src/anemoi/graphs/nodes/builders/from_mpas.py`
- Node builders for MPAS meshes (limited area and coarse)

#### `graphs/tests/edges/test_mpas_topological_edges.py`
- Tests for MPAS edge construction

#### `graphs/tests/nodes/test_limited_area_mpas_nodes.py`
- Tests for limited area MPAS nodes

#### `graphs/tests/nodes/test_mpas_coarse_nodes.py`
- Tests for coarse MPAS nodes

---

### 3. **Testing and Diagnostics**

#### `models/tests/preprocessing/test_residual_normalizer.py`
- Unit tests for residual normalizer

#### `training/src/anemoi/training/diagnostics/index_alignment.py`
- Diagnostic tool for checking data index alignment

#### `training/src/anemoi/training/diagnostics/run_index_alignment_audit.py`
- Script to run index alignment audit

#### `training/src/anemoi/training/train/timing_callbacks.py`
- Callbacks for timing different training phases

---

## Summary of Experiments

### 1. **WoFSCast vs Anemoi Implementation Comparison**
- Verified residual prediction frameworks match
- Confirmed normalization flow (physical space residual computation)
- Validated loss reduction (variable grouping by base name)
- Both predict normalized residuals: `Δx_norm = (x_{t+1} - x_t) / σ_Δx`

### 2. **Zero-Initialization Bug Fix**
- **Problem**: `initialise_data_extractor_zero: true` caused model to output all zeros
- **Root Cause**: Zero-init designed for skip-connection models, breaks residual-only models
- **Solution**: Changed to `initialise_data_extractor_zero: false`
- **Result**: Training stabilized, loss decreased from 6-8 to ~3.7

### 3. **Final LayerNorm Removal**
- **Rationale**: WoFSCast does not use LayerNorm before final projection
- **Implementation**: Added `final_layer_norm: false` option
- **Benefit**: Direct mapping from hidden → residual scales without normalization bottleneck

### 4. **Trajectory-Diverse Batching**
- **Purpose**: Ensure batches contain samples from different forecast trajectories
- **Method**: Round-robin sampling across trajectory IDs
- **Benefit**: Each batch sees diverse weather regimes, not dominated by single event

### 5. **Extreme Weather Handling**
- **Observation**: Training shows 115-sigma prediction errors (extreme outliers)
- **Recommendations**:
  - Huber loss with δ=1.0 to 3.0 (implemented)
  - Sample weighting to downweight extreme batches (implemented)
  - Gradient clipping (recommended: max_norm=10.0)

### 6. **Graph Connectivity for LAM**
- **Issue**: `margin_radius_km: 3` < grid spacing (3.75km)
- **Problem**: Hidden mesh barely extends beyond data grid, prevents boundary info flow
- **Recommendation**: Increase to 50km (~13 grid cells)

### 7. **Float32 Precision for Stability**
- All residual normalization done in float32 to avoid bfloat16 precision issues
- Cast back to original dtype for model compatibility

---

## Configuration Changes

### Critical Settings for Residual Model

```yaml
model:
  decoder:
    initialise_data_extractor_zero: false  # CRITICAL: Must be false for residual model
    final_layer_norm: false                 # Match WoFSCast (no LayerNorm before output)

graph:
  hidden:
    margin_radius_km: 50  # CRITICAL: Must be >> grid spacing (was 3, too small)

training:
  training_loss:
    _target_: anemoi.training.losses.GraphCastHuberLoss  # Recommended over MSE
    delta: 1.0  # Clips gradients from extreme errors
    scalers: ['node_weights']

  gradient_clip_val: 10.0  # Prevent gradient explosion from outliers
```

---

## Key Architectural Decisions

1. **Residual Prediction**: Model predicts normalized residuals, not next state
2. **Physical Space Computation**: Residuals computed in physical space before normalization
3. **No Skip Connections**: Skip removed from model, residual added in forecaster
4. **Variable Grouping**: Loss averages over vertical levels before summing groups
5. **Float32 Statistics**: All normalization/denormalization in float32 for stability
6. **Trajectory Diversity**: Batches contain samples from different weather events
7. **Huber Loss**: Robust to 100+ sigma outliers from extreme convection

---

## Current Training Status
- Step ~9798/175000 (6% through epoch 0)
- Loss: ~5.4 (down from initial 6-8)
- Observations: 115-sigma errors indicate need for Huber loss or graph connectivity fix
- No NaN/Inf issues
- Training stable at ~2.2 iterations/second

---

# Anemoi-Datasets Modifications

## Modified Files (4 files)

### 1. **Data Loading by Month**

#### `src/anemoi/datasets/data/dataset.py`
**Purpose**: Add capability to filter dataset by month(s)

**Changes**:
- **Lines 197-201**: Added month filtering in `_subset()` method
  - Checks if "months" kwarg present
  - Creates Subset with month-filtered indices
- **Lines 391-413**: Implemented `_months_to_indices()` method
  ```python
  def _months_to_indices(self, months: list[int] | int) -> list[int]:
      # Convert single int to list
      # Validate months are 1-12
      # Return indices where date.month in months_set
  ```
  - Accepts single month (int) or list of months
  - Validates month values (1-12)
  - Returns indices matching specified months

**Usage Example**:
```python
# Load only summer months (JJA)
dataset = dataset.subset(months=[6, 7, 8])

# Load only January
dataset = dataset.subset(months=1)
```

---

### 2. **Missing Data Handling**

#### `src/anemoi/datasets/data/stores.py`
**Purpose**: Fix missing data handling for forecast data with duplicate valid times

**Changes**:
- **Lines 459-471**: Updated `ZarrWithMissingDates.__init__()` to prefer `missing_indices` over `missing_dates`
  - **New behavior**: Checks for `missing_indices` attribute first
    - `missing_indices` stores actual dataset indices (correct for forecasts)
    - Avoids ambiguity when multiple forecast init times have same valid time
  - **Fallback**: Uses `missing_dates` if `missing_indices` not available
    - Legacy behavior for analysis data
    - Maps dates to indices (may be ambiguous for forecasts)

**Rationale**: Forecast datasets can have multiple samples with the same valid time (different initialization times). Using dates alone is ambiguous; indices are explicit.

**Example**:
```
Forecast A: Init 2024-01-01 00Z + 12h lead → Valid 2024-01-01 12Z (index 0)
Forecast B: Init 2024-01-01 06Z + 6h lead → Valid 2024-01-01 12Z (index 1)
```
If index 0 is missing, `missing_dates=['2024-01-01 12Z']` is ambiguous (both samples). `missing_indices=[0]` is explicit.

---

### 3. **Trajectory IDs for Diverse Batching**

#### `src/anemoi/datasets/data/subset.py`
**Purpose**: Enable trajectory-aware sampling in training

**Changes**:
- **Lines 283-287**: Added `trajectory_ids` property
  ```python
  @cached_property
  def trajectory_ids(self) -> NDArray[np.int64]:
      all_trajectory_ids = np.array(self.dataset.trajectory_ids)
      return all_trajectory_ids[self.indices]
  ```
  - Filters parent dataset's trajectory_ids by subset indices
  - Used by anemoi-training for trajectory-diverse batching
  - Each batch contains samples from different forecast trajectories

**Connection**: This property is consumed by `training/data/dataset.py:_get_trajectory_diverse_indices()` in anemoi-core to ensure batch diversity.

---

### 4. **Grid Visualization**

#### `src/anemoi/datasets/grids.py`
**Purpose**: Improve grid mask visualization with Cartopy for LAM domains

**Changes**:
- Complete rewrite of `plot_mask()` function (lines 27-185)
- **New features**:
  - Uses Cartopy for proper map projections
  - Lambert Conformal projection for regional zoomed views
  - PlateCarree for global views
  - Adds coastlines, borders, land features
  - Gridlines with labels
  - Higher resolution (150 DPI)
  - Automatic longitude normalization (-180 to 180)
  - Auto-centering for regional projection
  - Better legend and titles

- **Generated plots** (6 total):
  1. `path-global.png`: All global grid points
  2. `path-cutout.png`: Masked cutout points (global view)
  3. `path-lam.png`: LAM domain points (zoomed, Lambert projection)
  4. `path-both.png`: Global masked + LAM overlay (global view)
  5. `path-both-zoomed.png`: Global masked + LAM overlay (zoomed)
  6. `path-global-zoomed.png`: Global masked points (zoomed to LAM region)

**Before**: Simple scatter plots with plt.scatter, no map context
**After**: Professional cartographic plots with coastlines, borders, and proper projections

---

## Summary of anemoi-datasets Changes

### 1. **Month-Based Data Filtering**
- Enables seasonal analysis and training
- Filter by single month or list of months
- Example use cases:
  - Train on summer months only (convective season)
  - Validate on winter months (different weather regime)
  - Create seasonal climatologies

### 2. **Forecast Missing Data Fix**
- Correctly handles missing samples in forecast datasets
- Avoids ambiguity from duplicate valid times
- Explicit index-based missing data tracking

### 3. **Trajectory ID Support**
- Enables diverse batch sampling in training
- Subset operation preserves trajectory information
- Critical for preventing batch homogeneity

### 4. **Enhanced Grid Visualization**
- Professional-quality plots for LAM domains
- Uses proper map projections (Lambert Conformal)
- Geographic context (coastlines, borders)
- Multiple views for verification and debugging

---

## Complete Modification Overview

### Anemoi-Core
- **25 modified files**, **10 new files**
- Core focus: GraphCast-style residual prediction for storm-scale LAM
- Key changes: Skip connection removal, residual normalization, diagnostic logging, loss functions

### Anemoi-Datasets
- **4 modified files**, **0 new files**
- Core focus: Data filtering and quality-of-life improvements
- Key changes: Month filtering, missing data handling, trajectory IDs, grid visualization

### Total
- **29 modified files**, **10 new files** across both repos
- Comprehensive implementation of residual prediction framework
- Production-ready tools for storm-scale ML weather prediction
