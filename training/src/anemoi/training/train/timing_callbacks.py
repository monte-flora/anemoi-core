import time
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import logging

LOGGER = logging.getLogger(__name__)

class LightningOverheadProfiler(Callback):
    """Profile time spent in Lightning's control plane between batches."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.last_batch_end = None
        self.batch_received_from_dataloader = None
        self.batch_to_device_start = None
        self.training_step_start = None
        
        # Track timing components
        self.timings = {
            'lightning_control': [],  # Time between batches (outside user code)
            'dataloader_next': [],     # Actual DataLoader.__next__() time
            'batch_to_device': [],     # transfer_batch_to_device time
            'pre_training_step': [],   # Lightning hooks before training_step
        }
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called after batch is loaded but before training_step."""
        current_time = time.time()
        
        # Measure total inter-batch time (Lightning control plane)
        if self.last_batch_end is not None:
            total_gap = current_time - self.last_batch_end
            self.timings['lightning_control'].append(total_gap)
            
            if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
                LOGGER.info(f"[OVERHEAD] Total gap between batches: {total_gap:.4f}s")
        
        self.batch_received_from_dataloader = current_time
    
    def on_before_batch_transfer(self, trainer, pl_module, batch, dataloader_idx):
        """Called right after DataLoader.__next__() returns."""
        self.batch_to_device_start = time.time()
        
        if self.last_batch_end is not None:
            # This is the ACTUAL dataloader time
            dataloader_time = self.batch_to_device_start - self.last_batch_end
            self.timings['dataloader_next'].append(dataloader_time)
    
    def on_after_batch_transfer(self, trainer, pl_module, batch, dataloader_idx):
        """Called after batch moved to device."""
        if self.batch_to_device_start is not None:
            transfer_time = time.time() - self.batch_to_device_start
            self.timings['batch_to_device'].append(transfer_time)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after training_step, backward, and optimizer_step complete."""
        torch.cuda.synchronize()
        self.last_batch_end = time.time()
        
        # Log breakdown every N steps
        if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
            self._log_overhead_breakdown(batch_idx)
    
    def _log_overhead_breakdown(self, batch_idx):
        """Log detailed breakdown of where time is spent."""
        if len(self.timings['lightning_control']) == 0:
            return
        
        # Get recent averages
        n = min(self.log_every_n_steps, len(self.timings['lightning_control']))
        
        avg_total_gap = sum(self.timings['lightning_control'][-n:]) / n
        avg_dataloader = sum(self.timings['dataloader_next'][-n:]) / n if self.timings['dataloader_next'] else 0
        avg_transfer = sum(self.timings['batch_to_device'][-n:]) / n if self.timings['batch_to_device'] else 0
        
        # Calculate overhead (everything NOT in dataloader or transfer)
        lightning_overhead = avg_total_gap - avg_dataloader - avg_transfer
        
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Step {batch_idx} - Inter-Batch Overhead Breakdown:")
        LOGGER.info(f"  Total gap between batches: {avg_total_gap:.4f}s (100.0%)")
        LOGGER.info(f"  ├─ DataLoader.__next__():   {avg_dataloader:.4f}s ({avg_dataloader/avg_total_gap*100:5.1f}%)")
        LOGGER.info(f"  ├─ batch_to_device():       {avg_transfer:.4f}s ({avg_transfer/avg_total_gap*100:5.1f}%)")
        LOGGER.info(f"  └─ Lightning overhead:      {lightning_overhead:.4f}s ({lightning_overhead/avg_total_gap*100:5.1f}%)")
        
        # Diagnose specific issues
        if lightning_overhead > 0.1:
            LOGGER.warning(f"\n  ⚠️ HIGH LIGHTNING OVERHEAD! ({lightning_overhead:.4f}s)")
            LOGGER.warning(f"     Likely causes:")
            LOGGER.warning(f"     - val_dataloader() being called")
            LOGGER.warning(f"     - Dataset recreation")
            LOGGER.warning(f"     - Callback overhead")
            LOGGER.warning(f"     - Hook orchestration")
        
        if avg_dataloader > 0.1:
            LOGGER.warning(f"\n  ⚠️ DataLoader bottleneck! ({avg_dataloader:.4f}s)")
            LOGGER.warning(f"     This is actual I/O wait time")
        
        LOGGER.info(f"{'='*80}\n")


class DatasetRecreationDetector(Callback):
    """Detect if datasets are being recreated between batches."""
    
    def __init__(self):
        super().__init__()
        self.train_dataloader_id = None
        self.val_dataloader_id = None
        self.setup_call_count = 0
        self.train_dataloader_call_count = 0
        self.val_dataloader_call_count = 0
    
    def on_fit_start(self, trainer, pl_module):
        """Track initial dataloader IDs."""
        self.train_dataloader_id = id(trainer.train_dataloader)
        LOGGER.info(f"Initial train_dataloader ID: {self.train_dataloader_id}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Check if dataloader was recreated."""
        current_id = id(trainer.train_dataloader)
        if current_id != self.train_dataloader_id:
            LOGGER.warning(f"⚠️ train_dataloader RECREATED! Old ID: {self.train_dataloader_id}, New ID: {current_id}")
            self.train_dataloader_id = current_id
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Check if dataloader is being recreated between batches."""
        if batch_idx % 10 == 0:
            current_id = id(trainer.train_dataloader)
            if current_id != self.train_dataloader_id:
                LOGGER.warning(f"⚠️⚠️⚠️ train_dataloader RECREATED at step {batch_idx}!")
                LOGGER.warning(f"       Old ID: {self.train_dataloader_id}, New ID: {current_id}")
                self.train_dataloader_id = current_id
    
    def on_validation_start(self, trainer, pl_module):
        """Track when validation is triggered."""
        LOGGER.warning(f"🔍 VALIDATION STARTED at training step {trainer.global_step}")
        self.val_dataloader_call_count += 1


class CallbackOverheadProfiler(Callback):
    """Profile time spent in callbacks."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.callback_times = {}
        self.current_hook = None
        self.hook_start = None
    
    def _start_timing(self, hook_name):
        """Start timing a hook."""
        self.current_hook = hook_name
        self.hook_start = time.time()
    
    def _end_timing(self):
        """End timing current hook."""
        if self.current_hook and self.hook_start:
            elapsed = time.time() - self.hook_start
            if self.current_hook not in self.callback_times:
                self.callback_times[self.current_hook] = []
            self.callback_times[self.current_hook].append(elapsed)
            
            if elapsed > 0.01:  # Log if >10ms
                LOGGER.warning(f"Slow callback: {self.current_hook} took {elapsed:.4f}s")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_timing("on_train_batch_start")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_timing()
        
        if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
            LOGGER.info(f"\nCallback overhead summary:")
            for hook, times in self.callback_times.items():
                if times:
                    avg = sum(times[-self.log_every_n_steps:]) / min(self.log_every_n_steps, len(times))
                    if avg > 0.005:  # Only show if >5ms
                        LOGGER.info(f"  {hook}: {avg*1000:.2f}ms avg")


class DetailedTimingCallback(Callback):
    """Detailed timing breakdown for training steps."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.timings = {
            'data_loading': [],
            'forward': [],
            'loss_computation': [],
            'backward': [],
            'optimizer_step': [],
            'total': [],
        }
        
        # Timing markers
        self.batch_start = None
        self.forward_start = None
        self.backward_start = None
        self.optimizer_start = None
        self.last_batch_end = None
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        current_time = time.time()
        
        # Calculate data loading time (time between batches)
        if self.last_batch_end is not None:
            data_time = current_time - self.last_batch_end
            self.timings['data_loading'].append(data_time)
        
        self.batch_start = current_time
        self.forward_start = current_time
    
    def on_before_backward(self, trainer, pl_module, loss):
        torch.cuda.synchronize()
        forward_end = time.time()
        
        if self.forward_start is not None:
            forward_time = forward_end - self.forward_start
            self.timings['forward'].append(forward_time)
        
        self.backward_start = forward_end
    
    def on_after_backward(self, trainer, pl_module):
        torch.cuda.synchronize()
        backward_end = time.time()
        
        if self.backward_start is not None:
            backward_time = backward_end - self.backward_start
            self.timings['backward'].append(backward_time)
        
        self.optimizer_start = backward_end
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Optimizer start is already set in on_after_backward
        pass
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        batch_end = time.time()
        
        # Calculate optimizer step time
        if self.optimizer_start is not None:
            optimizer_time = batch_end - self.optimizer_start
            self.timings['optimizer_step'].append(optimizer_time)
        
        # Calculate total time
        if self.batch_start is not None:
            total_time = batch_end - self.batch_start
            self.timings['total'].append(total_time)
        
        self.last_batch_end = batch_end
        
        # Log every N steps
        if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
            self._log_timings(batch_idx)
    
    def _log_timings(self, batch_idx):
        """Log detailed timing breakdown."""
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Step {batch_idx} - Detailed Timing Breakdown:")
        
        # Calculate averages for recent steps
        n_recent = min(self.log_every_n_steps, len(self.timings['total']))
        
        if n_recent == 0:
            LOGGER.info("  No timing data available yet")
            LOGGER.info(f"{'='*80}\n")
            return
        
        def get_avg(key):
            if len(self.timings[key]) == 0:
                return 0.0
            recent = self.timings[key][-n_recent:]
            return sum(recent) / len(recent) if recent else 0.0
        
        avg_data = get_avg('data_loading')
        avg_forward = get_avg('forward')
        avg_backward = get_avg('backward')
        avg_optimizer = get_avg('optimizer_step')
        avg_total = get_avg('total')
        
        # Calculate percentages
        if avg_total > 0:
            pct_data = (avg_data / avg_total) * 100
            pct_forward = (avg_forward / avg_total) * 100
            pct_backward = (avg_backward / avg_total) * 100
            pct_optimizer = (avg_optimizer / avg_total) * 100
        else:
            pct_data = pct_forward = pct_backward = pct_optimizer = 0.0
        
        # Log breakdown
        LOGGER.info(f"  Total time:        {avg_total:.4f}s (100.0%)")
        LOGGER.info(f"  ├─ Data loading:   {avg_data:.4f}s ({pct_data:5.1f}%)")
        LOGGER.info(f"  ├─ Forward pass:   {avg_forward:.4f}s ({pct_forward:5.1f}%)")
        LOGGER.info(f"  ├─ Backward pass:  {avg_backward:.4f}s ({pct_backward:5.1f}%)")
        LOGGER.info(f"  └─ Optimizer step: {avg_optimizer:.4f}s ({pct_optimizer:5.1f}%)")
        LOGGER.info(f"")
        LOGGER.info(f"  Throughput: {1.0/avg_total:.4f} it/s")
        
        # Warnings
        if pct_data > 20:
            LOGGER.warning(f"  ⚠️  Data loading bottleneck! ({pct_data:.1f}%)")
            LOGGER.warning(f"     Consider: increase num_workers, prefetch_factor")
        
        if pct_backward > 50:
            LOGGER.warning(f"  ⚠️  Backward pass bottleneck! ({pct_backward:.1f}%)")
            LOGGER.warning(f"     Consider: reduce num_gpus, increase accum_grad_batches")
        
        if pct_optimizer > 30:
            LOGGER.warning(f"  ⚠️  Optimizer bottleneck! ({pct_optimizer:.1f}%)")
            LOGGER.warning(f"     Consider: use fused optimizer, check gradient clipping")
        

class DataLoaderTimingCallback(Callback):
    """Monitor dataloader performance."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.last_batch_end = None
        self.data_wait_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        current_time = time.time()
        
        if self.last_batch_end is not None:
            data_wait_time = current_time - self.last_batch_end
            self.data_wait_times.append(data_wait_time)
            
            if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
                avg_wait = sum(self.data_wait_times[-self.log_every_n_steps:]) / len(self.data_wait_times[-self.log_every_n_steps:])
                LOGGER.info(f"DataLoader wait time (last {self.log_every_n_steps} steps): {avg_wait:.4f}s avg")
                
                if avg_wait > 0.01:  # More than 10ms
                    LOGGER.warning(f"⚠️  DataLoader bottleneck detected! GPUs waiting {avg_wait:.4f}s for data")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        self.last_batch_end = time.time()


class DDPSyncTimingCallback(Callback):
    """Monitor DDP synchronization overhead."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.sync_times = []
        self.backward_start = None
        
    def on_before_backward(self, trainer, pl_module, loss):
        if dist.is_initialized():
            torch.cuda.synchronize()
            self.backward_start = time.time()
    
    def on_after_backward(self, trainer, pl_module):
        if dist.is_initialized() and self.backward_start is not None:
            torch.cuda.synchronize()
            sync_time = time.time() - self.backward_start
            self.sync_times.append(sync_time)
            
            batch_idx = trainer.global_step
            if batch_idx % self.log_every_n_steps == 0 and trainer.is_global_zero:
                avg_sync = sum(self.sync_times[-self.log_every_n_steps:]) / len(self.sync_times[-self.log_every_n_steps:])
                LOGGER.info(f"DDP backward+sync time: {avg_sync:.4f}s avg (rank {dist.get_rank()})")


class GPUUtilizationCallback(Callback):
    """Monitor per-GPU utilization to detect imbalance."""
    
    def __init__(self, log_every_n_steps=5):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                
                # Measure GPU memory
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                
                # Gather across all ranks
                mem_stats = torch.tensor([mem_allocated, mem_reserved], device='cuda')
                gathered = [torch.zeros(2, device='cuda') for _ in range(world_size)]
                dist.all_gather(gathered, mem_stats)
                
                if trainer.is_global_zero:
                    LOGGER.info(f"\n{'='*80}")
                    LOGGER.info(f"Step {batch_idx} - GPU Memory Usage:")
                    for r, stats in enumerate(gathered):
                        LOGGER.info(f"  GPU {r}: {stats[0]:.2f}GB allocated, {stats[1]:.2f}GB reserved")
                    
                    # Check for imbalance
                    allocated = [s[0].item() for s in gathered]
                    max_mem = max(allocated)
                    min_mem = min(allocated)
                    if max_mem > 0 and min_mem > 0:
                        imbalance = max_mem / min_mem
                        if imbalance > 1.2:
                            LOGGER.warning(f"⚠️  Memory imbalance detected: {imbalance:.2f}x")
                    LOGGER.info(f"{'='*80}\n")