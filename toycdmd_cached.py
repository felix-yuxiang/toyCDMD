"""
CDMD with Cached Trajectories for 2D Toy Datasets (Flow Matching Version)

This implements the cached trajectory algorithm adapted for flow matching models.
Key differences from EDM diffusion:
- Uses flow matching interpolation: x_t = (1-t) * x_0 + t * noise
- Velocity field v(x, t) predicts: noise - x_0
- Time goes from t=1 (noise) to t=0 (data)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

# Import shared components from toycdmd
from toycdmd import (
    GaussianGenerator,
    CrescentGenerator,
    SpiralGenerator,
    CheckerboardGenerator,
    VelocityNet,
    forward_euler_integration,
    forward_euler_integration_no_grad,
    load_teacher_model,
)


class TrajectoryCache:
    """
    Cache for storing trajectory points during CDMD training.

    Each trajectory stores points along the ODE path from noise (t=1) to data (t=0).
    For flow matching: x_t = (1-t) * x_0 + t * noise
    """

    def __init__(
        self,
        num_trajectories: int,
        n_dims: int,
        max_steps: int,
        source_generator,
        device: str = 'cpu'
    ):
        """
        Args:
            num_trajectories: Number of trajectories in the cache (K)
            n_dims: Dimension of data points
            max_steps: Maximum number of steps per trajectory (T)
            source_generator: Generator for noise samples
            device: Device to store tensors
        """
        self.num_trajectories = num_trajectories
        self.n_dims = n_dims
        self.max_steps = max_steps
        self.source_generator = source_generator
        self.device = device

        # Cache storage: list of trajectories, each trajectory is a list of points
        # C[i] stores points [x_T, x_{T-1}, ..., x_0] for trajectory i
        # In flow matching terms: [x_{t=1}, x_{t=1-dt}, ..., x_{t=0}]
        self.trajectories: List[List[torch.Tensor]] = []

        # Lifespan counters for each trajectory
        self.lifespan_counters = torch.zeros(num_trajectories, dtype=torch.long, device=device)

        # Initialize with noise samples
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache with noise samples (one point per trajectory)."""
        self.trajectories = []
        # Generate all noise samples at once using source_generator
        all_noise = self.source_generator.generate(self.num_trajectories).to(self.device)
        for i in range(self.num_trajectories):
            # Start each trajectory with a noise sample (t=1)
            self.trajectories.append([all_noise[i]])

    def get_trajectory_length(self, idx: int) -> int:
        """Get current length of trajectory idx."""
        return len(self.trajectories[idx])

    def is_complete(self, idx: int) -> bool:
        """Check if trajectory idx is complete (reached max_steps)."""
        return len(self.trajectories[idx]) >= self.max_steps

    def reset_trajectory(self, idx: int):
        """Reset trajectory idx with new noise sample."""
        noise = self.source_generator.generate(1).to(self.device).squeeze(0)
        self.trajectories[idx] = [noise]
        self.lifespan_counters[idx] = 0

    def get_initial_noise(self, idx: int) -> torch.Tensor:
        """Get the initial noise (x_T, t=1) for trajectory idx."""
        return self.trajectories[idx][0]

    def get_point_at_step(self, idx: int, step: int) -> torch.Tensor:
        """Get point at given step for trajectory idx."""
        return self.trajectories[idx][step]

    def append_point(self, idx: int, point: torch.Tensor):
        """Append a new point to trajectory idx."""
        self.trajectories[idx].append(point.detach().clone())

    def increment_lifespan(self, idx: int):
        """Increment lifespan counter for trajectory idx."""
        self.lifespan_counters[idx] += 1

    def get_lifespan(self, idx: int) -> int:
        """Get lifespan counter for trajectory idx."""
        return self.lifespan_counters[idx].item()


def get_timestep_from_step(step: int, max_steps: int) -> float:
    """
    Convert step index to timestep value.

    For flow matching:
    - step=0 corresponds to t=1 (pure noise)
    - step=max_steps-1 corresponds to t=0 (pure data)

    Returns timestep t in [0, 1].
    """
    # Linear schedule from t=1 to t=0
    return 1.0 - step / (max_steps - 1)


def compute_cached_cdmd_loss(
    cache: TrajectoryCache,
    student_model: nn.Module,
    teacher_model: nn.Module,
    batch_indices: torch.Tensor,
    max_lifespan: int,
    student_steps: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CDMD loss using cached trajectories.

    Args:
        cache: Trajectory cache
        student_model: Student model to train
        teacher_model: Pre-trained teacher (frozen)
        batch_indices: Indices of trajectories to use in this batch
        max_lifespan: Maximum lifespan before resetting trajectory (J_max)
        student_steps: Number of steps for student generation
        device: Device to use

    Returns:
        Tuple of (loss, generated_samples)
    """
    batch_size = len(batch_indices)
    max_steps = cache.max_steps

    all_x_theta = []
    all_mu_f = []
    all_mu_r = []
    all_weights = []

    for i, traj_idx in enumerate(batch_indices):
        traj_idx = traj_idx.item()

        # Check if trajectory is complete and handle lifespan
        if cache.is_complete(traj_idx):
            if cache.get_lifespan(traj_idx) > max_lifespan:
                # Reset old trajectory
                cache.reset_trajectory(traj_idx)
            else:
                cache.increment_lifespan(traj_idx)

        # Sample step s from [0, |C_i| - 1]
        current_length = cache.get_trajectory_length(traj_idx)
        s = torch.randint(0, current_length, (1,)).item()

        # Get timestep tau = T - s (in our case, t value)
        t_tau = get_timestep_from_step(s, max_steps)
        t_tau_tensor = torch.tensor([[t_tau]], device=device)

        # Get x_T (initial noise) and x_tau (point at step s)
        x_T = cache.get_initial_noise(traj_idx).unsqueeze(0).to(device)
        x_tau = cache.get_point_at_step(traj_idx, s).unsqueeze(0).to(device)

        # Sample new noise z for forward mapping
        z = torch.randn_like(x_T).to(device)

        # Student forward pass: generate x_theta from noise
        x_theta = forward_euler_integration(
            x_T, student_model, t_start=1.0, t_end=0.0,
            num_steps=student_steps, device=device
        )
        all_x_theta.append(x_theta)

        # Cache management: handle based on whether we're at the latest timestep
        is_latest_step = (s == current_length - 1)
        can_extend = (s < max_steps - 1)

        if is_latest_step:
            # At the latest timestep of this trajectory
            # Get teacher prediction at current point
            with torch.no_grad():
                v_tau = teacher_model(x_tau, t_tau_tensor)

            if can_extend:
                # Extend the trajectory by computing next point
                t_tau_minus_1 = get_timestep_from_step(s + 1, max_steps)
                dt = t_tau - t_tau_minus_1  # positive since t decreases

                # Accurate ODE reverse step: x_{tau-1} = x_tau - dt * v(x_tau, t_tau)
                with torch.no_grad():
                    x_tau_minus_1 = x_tau - dt * v_tau
                    # Append to cache
                    cache.append_point(traj_idx, x_tau_minus_1.squeeze(0))

            # Use teacher velocity at current point as reference
            mu_r = x_tau - v_tau * t_tau_tensor
        else:
            # Interim timestep - we have cached point at s+1
            x_tau_minus_1 = cache.get_point_at_step(traj_idx, s + 1).unsqueeze(0).to(device)
            t_tau_minus_1 = get_timestep_from_step(s + 1, max_steps)

            # Estimate x_0 from two cached points using linear interpolation
            # x_tau = (1-t_tau) * x_0 + t_tau * noise
            # x_{tau-1} = (1-t_{tau-1}) * x_0 + t_{tau-1} * noise
            if t_tau_minus_1 != t_tau:
                x_0_hat = x_tau + (x_tau_minus_1 - x_tau) / (t_tau_minus_1 - t_tau) * (0 - t_tau)
            else:
                x_0_hat = x_tau

            # Apply forward mapping: x_t = (1-t) * x_0 + t * z where z is new noise
            x_noisy = (1 - t_tau) * x_0_hat + t_tau * z

            with torch.no_grad():
                v_r = teacher_model(x_noisy, t_tau_tensor)
                mu_r = x_noisy - v_r * t_tau_tensor

        # Coarse sample 1-step denoising from student
        # Apply forward mapping: x_t = (1-t) * x_0 + t * z where z is new noise
        x_theta_noisy = (1 - t_tau) * x_theta + t_tau * z

        with torch.no_grad():
            v_f = teacher_model(x_theta_noisy, t_tau_tensor)
            mu_f = x_theta_noisy - v_f * t_tau_tensor

        all_mu_f.append(mu_f)
        all_mu_r.append(mu_r)

        # Weight factor
        # weight = ``.0 / max(t_tau, 0.01)  # Higher weight for smaller t
        # weight = 1.0 / torch.abs(x_theta - mu_r).mean(dim=1, keepdim=True).clamp(min=1e-4)
        # weight = 1.0 
        if torch.sum(x_theta - mu_r) == 0.0:
            weight = torch.tensor([[1.0]], device=device)
        else:
            weight = 1.0 / torch.abs(x_theta - mu_r).mean(dim=1, keepdim=True).clamp(min=1e-4)

        all_weights.append(weight) 

    # Stack all tensors
    x_theta_batch = torch.cat(all_x_theta, dim=0)
    mu_f_batch = torch.cat(all_mu_f, dim=0)
    mu_r_batch = torch.cat(all_mu_r, dim=0)
    weights = torch.cat(all_weights, dim=0) 

    # Compute gradient: grad = w(t) * (mu_f - mu_r)
    grad = weights * (mu_f_batch - mu_r_batch)
    # grad = (mu_f_batch - mu_r_batch)
    grad = torch.nan_to_num(grad)

    # CDM loss
    loss = 0.5 * F.mse_loss(x_theta_batch, (x_theta_batch - grad).detach())

    return loss, x_theta_batch


def train_cdmd_cached(
    teacher_model: nn.Module,
    student_model: nn.Module,
    source_generator,
    num_trajectories: int = 1024,
    max_steps: int = 10,
    max_lifespan: int = 5,
    n_epochs: int = 100,
    lr_student: float = 1e-4,
    batch_size: int = 256,
    batches_per_epoch: int = 50,
    student_steps: int = 1,
    epoch_save_freq: int = 10,
    checkpoint_prefix: str = 'cdmd_cached_model',
    device: str = 'cpu'
):
    """
    CDMD training loop with cached trajectories.

    Args:
        teacher_model: Pre-trained teacher (frozen)
        student_model: Student model to train
        source_generator: Noise generator
        num_trajectories: Number of trajectories in cache (K)
        max_steps: Maximum steps per trajectory (T)
        max_lifespan: Maximum lifespan before reset (J_max)
        n_epochs: Number of training epochs
        lr_student: Learning rate for student
        batch_size: Batch size
        batches_per_epoch: Batches per epoch
        student_steps: Number of integration steps for student
        epoch_save_freq: Save frequency
        checkpoint_prefix: Checkpoint prefix
        device: Device to use

    Returns:
        Trained student model
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize trajectory cache
    cache = TrajectoryCache(
        num_trajectories=num_trajectories,
        n_dims=2,
        max_steps=max_steps,
        source_generator=source_generator,
        device=device
    )

    optimizer_student = optim.Adam(student_model.parameters(), lr=lr_student)

    step = 0
    for epoch in range(n_epochs):
        student_model.train()
        total_loss = 0.0

        for batch_idx in range(batches_per_epoch):
            # Sample trajectory indices uniformly
            batch_indices = torch.randint(0, num_trajectories, (batch_size,), device=device)

            optimizer_student.zero_grad()

            loss, _ = compute_cached_cdmd_loss(
                cache=cache,
                student_model=student_model,
                teacher_model=teacher_model,
                batch_indices=batch_indices,
                max_lifespan=max_lifespan,
                student_steps=student_steps,
                device=device
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer_student.step()

            total_loss += loss.item()
            step += 1

        avg_loss = total_loss / batches_per_epoch

        # Count complete trajectories
        complete_count = sum(1 for i in range(num_trajectories) if cache.is_complete(i))

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}, Complete trajectories: {complete_count}/{num_trajectories}")

        if epoch % epoch_save_freq == 0:
            checkpoint_path = f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_model.state_dict(),
                'optimizer_student_state_dict': optimizer_student.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    checkpoint_path = f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'student_state_dict': student_model.state_dict(),
        'optimizer_student_state_dict': optimizer_student.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")

    return student_model


def visualize_cached_cdmd_results(
    student_model: nn.Module,
    teacher_model: nn.Module,
    source_generator,
    target_generator,
    num_points: int = 4000,
    student_steps: int = 1,
    teacher_steps: int = 100,
    device: str = 'cpu',
    save_path: str = None
):
    """Visualize CDMD results comparing student and teacher."""
    student_model.eval()
    teacher_model.eval()

    noise = source_generator.generate(num_points).to(device)
    target_points = target_generator.generate(num_points).cpu().numpy()

    # Student generation
    student_samples = forward_euler_integration_no_grad(
        noise, student_model, t_start=1.0, t_end=0.0,
        num_steps=student_steps, device=device
    ).cpu().numpy()

    # Teacher generation
    teacher_samples = forward_euler_integration_no_grad(
        noise, teacher_model, t_start=1.0, t_end=0.0,
        num_steps=teacher_steps, device=device
    ).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Target distribution
    axes[0].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.3, color='blue')
    axes[0].set_title('Target Distribution', fontsize=14)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Teacher generation
    axes[1].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.1, color='blue', label='Target')
    axes[1].scatter(teacher_samples[:, 0], teacher_samples[:, 1], s=5, alpha=0.3, color='green', label='Teacher')
    axes[1].set_title(f'Teacher ({teacher_steps} steps)', fontsize=14)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Student generation
    axes[2].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.1, color='blue', label='Target')
    axes[2].scatter(student_samples[:, 0], student_samples[:, 1], s=5, alpha=0.3, color='red', label='Student')
    axes[2].set_title(f'Student ({student_steps} step{"s" if student_steps > 1 else ""})', fontsize=14)
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cached CDMD training for 2D toy datasets')
    parser.add_argument('--teacher_checkpoint', type=str,
                        default='./models/flow_model_spiral_epoch_100.pt',
                        help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--dataset', type=str, default='spiral',
                        choices=['checkerboard', 'crescent', 'spiral'],
                        help='Target dataset (should match teacher training)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of CDMD training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr_student', type=float, default=1e-4,
                        help='Learning rate for student model')
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--student_steps', type=int, default=1,
                        help='Number of integration steps for student')
    parser.add_argument('--num_trajectories', type=int, default=1024,
                        help='Number of trajectories in cache (K)')
    parser.add_argument('--max_steps', type=int, default=10,
                        help='Maximum steps per trajectory (T)')
    parser.add_argument('--max_lifespan', type=int, default=5,
                        help='Maximum lifespan before reset (J_max)')
    parser.add_argument('--epoch_save_freq', type=int, default=10,
                        help='Save checkpoint frequency')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_points', type=int, default=4000,
                        help='Number of points for visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Source generator (Gaussian noise)
    source_generator = GaussianGenerator(n_dims=2, noise_std=1.0)

    # Target generator (for visualization)
    if args.dataset == 'checkerboard':
        target_generator = CheckerboardGenerator(grid_size=4, scale=1.5)
    elif args.dataset == 'crescent':
        target_generator = CrescentGenerator(R=1.0, r=0.6, d=0.5)
    elif args.dataset == 'spiral':
        target_generator = SpiralGenerator(noise_std=0.05, n_turns=3, radius_scale=2)

    # Load pre-trained teacher model
    teacher_model = load_teacher_model(
        args.teacher_checkpoint,
        input_dim=2,
        h_dim=args.h_dim,
        device=device
    )

    # Initialize student model from teacher
    student_model = VelocityNet(input_dim=2, h_dim=args.h_dim).to(device)
    student_model.load_state_dict(teacher_model.state_dict())
    print("Initialized student model from teacher checkpoint")

    checkpoint_prefix = f'{args.output_dir}/cdmd_cached_{args.dataset}_steps{args.student_steps}'

    print(f"\nStarting Cached CDMD training:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Student steps: {args.student_steps}")
    print(f"  Num trajectories (K): {args.num_trajectories}")
    print(f"  Max steps per trajectory (T): {args.max_steps}")
    print(f"  Max lifespan (J_max): {args.max_lifespan}")
    print(f"  Epochs: {args.epochs}")

    start_time = time.time()

    student_model = train_cdmd_cached(
        teacher_model=teacher_model,
        student_model=student_model,
        source_generator=source_generator,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
        max_lifespan=args.max_lifespan,
        n_epochs=args.epochs,
        lr_student=args.lr_student,
        batch_size=args.batch_size,
        batches_per_epoch=50,
        student_steps=args.student_steps,
        epoch_save_freq=args.epoch_save_freq,
        checkpoint_prefix=checkpoint_prefix,
        device=device
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining took {elapsed_time:.2f} seconds")

    # Visualize results
    print("\nGenerating visualization...")
    visualize_cached_cdmd_results(
        student_model=student_model,
        teacher_model=teacher_model,
        source_generator=source_generator,
        target_generator=target_generator,
        num_points=args.num_points,
        student_steps=args.student_steps,
        teacher_steps=100,
        device=device,
        save_path=f'./imgs/cdmd_cached_{args.dataset}_steps{args.student_steps}.png'
    )
