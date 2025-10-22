import logging
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from anemoi.training.losses.mse import MSELoss

LOGGER = logging.getLogger(__name__)

class RMSELossNormalized(MSELoss):
    """RMSE normalized by the mean target value, expressed as a percentage."""

    name: str = "normalized_rmse_pct"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """
        Calculates the RMSE normalized by the mean target value (% of mean).

        Returns
        -------
        torch.Tensor
            Normalized RMSE (% of mean target)
        """
        # Compute mean squared error using the parent class
        mse = super().forward(
            pred=pred,
            target=target,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
        )

        rmse = torch.sqrt(mse)

        # Mean of the target (absolute value to handle sign)
        mean_val = torch.mean(torch.abs(target))
        mean_val = torch.clamp(mean_val, min=1e-8)  # avoid division by zero

        # Express as a percentage of mean target
        normalized_rmse_pct = (rmse / mean_val) * 100.0

        return normalized_rmse_pct
