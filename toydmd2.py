"""
DMD2 (Distribution Matching Distillation 2) for 2D Toy Datasets

Reference: https://github.com/tianweiy/DMD2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


class GaussianGenerator:
    def __init__(self, n_dims=2, noise_std=1.0):
        self.n_dims = n_dims
        self.noise_std = noise_std

    def generate(self, num_points):
        return torch.randn(num_points, self.n_dims) * self.noise_std


class CrescentGenerator:
    def __init__(self, R=1.0, r=0.6, d=0.5):
        self.R = R
        self.r = r
        self.d = d

    def generate(self, num_points):
        outer_area = np.pi * self.R**2
        inner_area = np.pi * self.r**2
        crescent_area = outer_area - inner_area
        n_samples = int(num_points * (outer_area / crescent_area) * 1.2)
        n_samples = max(n_samples, num_points)

        theta = 2 * np.pi * torch.rand(n_samples)
        radius = self.R * torch.sqrt(torch.rand(n_samples))
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        mask = (x - self.d)**2 + y**2 > self.r**2
        points = torch.stack((x[mask], y[mask]), dim=1)

        while points.shape[0] < num_points:
            additional_points = self.generate(num_points - points.shape[0])
            points = torch.cat((points, additional_points), dim=0)

        return points[:num_points].to(dtype=torch.float32)


class SpiralGenerator:
    def __init__(self, noise_std=0.1, n_turns=4, radius_scale=0.5):
        self.noise_std = noise_std
        self.n_turns = n_turns
        self.radius_scale = radius_scale

    def generate(self, num_points):
        max_angle = 2 * np.pi * self.n_turns
        t = torch.linspace(0, max_angle, num_points)
        t = t * torch.pow(torch.rand(num_points), 0.5)

        r = self.radius_scale * (t / max_angle + 0.1)
        x = r * torch.cos(t)
        y = r * torch.sin(t)

        x += torch.randn(num_points) * self.noise_std
        y += torch.randn(num_points) * self.noise_std
        return torch.stack([x, y], dim=1)


class CheckerboardGenerator:
    def __init__(self, grid_size=3, scale=5.0, device='cpu'):
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    def generate(self, num_points):
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)

        while samples.shape[0] < num_points:
            new_samples = (torch.rand(num_points, 2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
            y_mask = torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)

        return samples[:num_points]


class VelocityNet(nn.Module):
    """Velocity network for flow matching"""
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.fc_in = nn.Linear(input_dim + 1, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)


def forward_euler_integration(
    initial_points: torch.Tensor,
    model: nn.Module,
    t_start: float = 1.0,
    t_end: float = 0.0,
    num_steps: int = 100,
    device='cpu'
) -> torch.Tensor:
    """Forward Euler integration for flow matching (with gradients)"""
    dt = (t_start - t_end) / num_steps
    current_points = initial_points.clone().to(device)

    for step in range(num_steps):
        current_time = t_start - step * dt
        t_tensor = torch.full((len(current_points), 1), current_time, device=device)
        velocity = model(current_points, t_tensor)
        current_points = current_points - velocity * dt

    return current_points


@torch.no_grad()
def forward_euler_integration_no_grad(
    initial_points: torch.Tensor,
    model: nn.Module,
    t_start: float = 1.0,
    t_end: float = 0.0,
    num_steps: int = 100,
    device='cpu'
) -> torch.Tensor:
    """Forward Euler integration without gradients (for visualization)"""
    dt = (t_start - t_end) / num_steps
    current_points = initial_points.clone().to(device)

    for step in range(num_steps):
        current_time = t_start - step * dt
        t_tensor = torch.full((len(current_points), 1), current_time, device=device)
        velocity = model(current_points, t_tensor)
        current_points = current_points - velocity * dt

    return current_points


def compute_distribution_matching_loss(
    noise: torch.Tensor,
    student_model: nn.Module,
    teacher_model: nn.Module,
    proxy_model: nn.Module,
    student_steps: int,
    t: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute DMD2 distribution matching loss.

    The key idea:
    - Student generates samples from noise in few steps
    - Teacher and proxy models predict velocities at interpolated points
    - Loss encourages student to match teacher's distribution via proxy guidance

    Args:
        noise: Input noise samples (batch_size, dim)
        student_model: Student model (generates in few steps)
        teacher_model: Pre-trained teacher (frozen)
        proxy_model: Proxy model (learns fake distribution)
        student_steps: Number of steps for student generation
        t: Random timesteps for interpolation
        device: Device to use

    Returns:
        Distribution matching loss
    """
    batch_size = noise.shape[0]

    # Student generates samples from noise (with gradients)
    generated_samples = forward_euler_integration(
        noise, student_model, t_start=1.0, t_end=0.0,
        num_steps=student_steps, device=device
    )

    # Create interpolated samples: x_t = (1-t) * x_generated + t * noise
    # This follows flow matching interpolation convention
    noise = torch.randn_like(noise).to(device)
    x_t = (1 - t) * generated_samples + t * noise

    # Get velocity predictions from teacher (frozen) and proxy
    with torch.no_grad():
        teacher_velocity = teacher_model(x_t, t)
        proxy_velocity = proxy_model(x_t, t)

        # Distribution matching loss: difference between teacher and proxy predictions
        p_diff = (teacher_velocity - proxy_velocity) * t
        weight_factor = torch.abs(generated_samples - x_t + t * teacher_velocity).mean(dim=1, keepdim=True)
        # p_diff = (teacher_velocity - proxy_velocity) 
        # weight_factor = torch.abs(teacher_velocity).mean(dim=1, keepdim=True).clamp(min=1e-4)
        grad = p_diff / weight_factor
        grad = torch.nan_to_num(grad)
        # grad = p_diff

    # The loss encourages generated samples to move towards teacher distribution
    loss_dm = 0.5 * F.mse_loss(generated_samples, (generated_samples - grad).detach())

    return loss_dm, generated_samples


def compute_proxy_loss(
    generated_samples: torch.Tensor,
    proxy_model: nn.Module,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Train proxy model to estimate velocity on generated (fake) samples.

    Args:
        generated_samples: Samples generated by student (detached)
        noise: Original noise used for generation
        proxy_model: Proxy model to train
        device: Device to use

    Returns:
        Proxy model loss
    """
    batch_size = generated_samples.shape[0]
    t = torch.rand(batch_size, 1, device=device)
    noise = torch.randn_like(generated_samples).to(device)

    # Interpolate between generated samples and noise
    interpolated = (1 - t) * generated_samples.detach() + t * noise

    # Target velocity: noise - generated_samples (flow from generated to noise)
    target_velocity = noise - generated_samples.detach()

    # Proxy predicts velocity
    proxy_velocity = proxy_model(interpolated, t)

    loss_proxy = F.mse_loss(proxy_velocity, target_velocity)

    return loss_proxy


def train_dmd2(
    teacher_model: nn.Module,
    proxy_model: nn.Module,
    student_model: nn.Module,
    source_generator,
    n_epochs: int = 100,
    lr_student: float = 1e-4,
    lr_proxy: float = 1e-4,
    batch_size: int = 2048,
    batches_per_epoch: int = 50,
    student_steps: int = 1,
    proxy_update_ratio: int = 1,
    epoch_save_freq: int = 10,
    checkpoint_prefix: str = 'dmd2_model',
    device: str = 'cpu'
):
    """
    DMD2 training loop

    Uses two time-scale update rule:
    - Proxy model is updated more frequently to track student distribution
    - Student model is updated to match teacher distribution via proxy guidance

    Args:
        teacher_model: Pre-trained teacher (frozen)
        proxy_model: Proxy model for fake distribution
        student_model: Student model to train
        source_generator: Noise generator
        n_epochs: Number of training epochs
        lr_student: Learning rate for student
        lr_proxy: Learning rate for proxy
        batch_size: Batch size
        batches_per_epoch: Batches per epoch
        student_steps: Number of integration steps for student
        proxy_update_ratio: How many proxy updates per student update
        epoch_save_freq: Save frequency
        checkpoint_prefix: Checkpoint prefix
        device: Device to use

    Returns:
        Trained student and proxy models
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer_student = optim.Adam(student_model.parameters(), lr=lr_student)
    optimizer_proxy = optim.Adam(proxy_model.parameters(), lr=lr_proxy)

    step = 0
    for epoch in range(n_epochs):
        student_model.train()
        proxy_model.train()

        total_loss_dm = 0.0
        total_loss_proxy = 0.0

        for batch_idx in range(batches_per_epoch):
            noise = source_generator.generate(batch_size).to(device)
            t = torch.rand(batch_size, 1, device=device)

            # Update student model
            optimizer_student.zero_grad()

            loss_dm, _ = compute_distribution_matching_loss(
                noise, student_model, teacher_model, proxy_model,
                student_steps, t, device
            )
            loss_dm.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer_student.step()


              # Update proxy model (more frequently for two time-scale)
            for _ in range(proxy_update_ratio):
                optimizer_proxy.zero_grad()

                # Generate samples with student (no grad for proxy update)
                with torch.no_grad():
                    generated_samples = forward_euler_integration_no_grad(
                        noise, student_model, t_start=1.0, t_end=0.0,
                        num_steps=student_steps, device=device
                    )

                loss_proxy = compute_proxy_loss(
                    generated_samples, proxy_model, device
                )
                loss_proxy.backward()
                optimizer_proxy.step()

            total_loss_dm += loss_dm.item()
            total_loss_proxy += loss_proxy.item()
            step += 1

        avg_loss_dm = total_loss_dm / batches_per_epoch
        avg_loss_proxy = total_loss_proxy / batches_per_epoch
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss DM: {avg_loss_dm:.6f}, Loss Proxy: {avg_loss_proxy:.6f}")

        if epoch % epoch_save_freq == 0:
            checkpoint_path = f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_model.state_dict(),
                'proxy_state_dict': proxy_model.state_dict(),
                'optimizer_student_state_dict': optimizer_student.state_dict(),
                'optimizer_proxy_state_dict': optimizer_proxy.state_dict(),
                'loss_dm': avg_loss_dm,
                'loss_proxy': avg_loss_proxy,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    checkpoint_path = f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'student_state_dict': student_model.state_dict(),
        'proxy_state_dict': proxy_model.state_dict(),
        'optimizer_student_state_dict': optimizer_student.state_dict(),
        'optimizer_proxy_state_dict': optimizer_proxy.state_dict(),
        'loss_dm': avg_loss_dm,
        'loss_proxy': avg_loss_proxy,
    }, checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")

    return student_model, proxy_model


def visualize_dmd2_results(
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
    """Visualize DMD2 results comparing student and teacher"""
    student_model.eval()
    teacher_model.eval()

    noise = source_generator.generate(num_points).to(device)
    target_points = target_generator.generate(num_points).cpu().numpy()

    # Student generation (few steps)
    student_samples = forward_euler_integration_no_grad(
        noise, student_model, t_start=1.0, t_end=0.0,
        num_steps=student_steps, device=device
    ).cpu().numpy()

    # Teacher generation (many steps)
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


def load_teacher_model(checkpoint_path: str, input_dim: int = 2, h_dim: int = 128, device: str = 'cpu'):
    """Load pre-trained teacher model from checkpoint"""
    model = VelocityNet(input_dim=input_dim, h_dim=h_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded teacher model from {checkpoint_path}")
    return model


def load_dmd2_student_model(checkpoint_path: str, input_dim: int = 2, h_dim: int = 128, device: str = 'cpu'):
    """Load trained DMD2 student model from checkpoint"""
    model = VelocityNet(input_dim=input_dim, h_dim=h_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['student_state_dict'])
    model.eval()
    print(f"Loaded student model from {checkpoint_path}")
    return model


def visualize_dmd2_progress(
    n_epochs: int,
    initial_points: torch.Tensor,
    teacher_model: nn.Module,
    target_points: np.ndarray,
    student_steps_list: list,
    teacher_steps: int,
    model_kwargs: dict,
    checkpoint_prefix: str,
    epoch_step: int = 10,
    device: str = 'cpu',
    suptitle: str = "DMD2 Training Progress"
):
    """
    Visualizes DMD2 student model results across training epochs.

    Args:
        n_epochs: Total number of training epochs.
        initial_points: Noisy input points (torch.Tensor).
        teacher_model: Pre-trained teacher model.
        target_points: Target distribution (numpy array).
        student_steps_list: List of step counts for student generation.
        teacher_steps: Number of steps for teacher generation.
        model_kwargs: Dict of kwargs to initialize model.
        checkpoint_prefix: Prefix of checkpoint files.
        epoch_step: Interval at which to visualize checkpoints.
        device: Torch device.
        suptitle: Title for the entire figure.
    """
    steps = list(range(0, n_epochs + 1, epoch_step))
    n_rows = len(steps)
    n_cols = len(student_steps_list) + 2  # +1 for teacher, +1 for proxy
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Generate teacher samples once (reference)
    teacher_samples = forward_euler_integration_no_grad(
        initial_points, teacher_model, t_start=1.0, t_end=0.0,
        num_steps=teacher_steps, device=device
    ).cpu().numpy()

    for i, epoch in enumerate(steps):
        model_name = f'{checkpoint_prefix}_epoch_{epoch}.pt'
        print(f"Loading model {model_name}")

        # Load student and proxy models
        student_model = VelocityNet(**model_kwargs).to(device)
        proxy_model = VelocityNet(**model_kwargs).to(device)
        checkpoint = torch.load(model_name, map_location=device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        proxy_model.load_state_dict(checkpoint['proxy_state_dict'])
        student_model.eval()
        proxy_model.eval()

        # Teacher reference (first column)
        ax = axes[i, 0]
        ax.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax.scatter(teacher_samples[:, 0], teacher_samples[:, 1], s=5, alpha=0.3, color='green', label='Teacher')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title(f'Teacher ({teacher_steps} steps)', fontsize=16)
        ax.set_ylabel(f"Epoch {epoch}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Proxy model (second column) - runs with 100 steps like teacher
        proxy_samples = forward_euler_integration_no_grad(
            initial_points, proxy_model, t_start=1.0, t_end=0.0,
            num_steps=teacher_steps, device=device
        ).cpu().numpy()

        ax = axes[i, 1]
        ax.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax.scatter(proxy_samples[:, 0], proxy_samples[:, 1], s=5, alpha=0.3, color='orange', label='Proxy')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title(f'Proxy ({teacher_steps} steps)', fontsize=16)
        ax.grid(True, alpha=0.3)

        # Student with different step counts
        for j, num_steps in enumerate(student_steps_list):
            student_samples = forward_euler_integration_no_grad(
                initial_points, student_model, t_start=1.0, t_end=0.0,
                num_steps=num_steps, device=device
            ).cpu().numpy()

            ax = axes[i, j + 2]
            ax.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
            ax.scatter(student_samples[:, 0], student_samples[:, 1], s=5, alpha=0.3, color='red', label='Student')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'Student ({num_steps} step{"s" if num_steps > 1 else ""})', fontsize=16)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=20, y=1.02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMD2 training for 2D toy datasets (without GAN loss)')
    parser.add_argument('--teacher_checkpoint', type=str,
                        default='./models/flow_model_spiral_epoch_100.pt',
                        help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--dataset', type=str, default='spiral',
                        choices=['checkerboard', 'crescent', 'spiral'],
                        help='Target dataset (should match teacher training)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of DMD2 training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr_student', type=float, default=1e-4,
                        help='Learning rate for student model')
    parser.add_argument('--lr_proxy', type=float, default=1e-4,
                        help='Learning rate for proxy model')
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--student_steps', type=int, default=1,
                        help='Number of integration steps for student (1 for single-step)')
    parser.add_argument('--proxy_update_ratio', type=int, default=1,
                        help='Proxy updates per student update (two time-scale)')
    parser.add_argument('--epoch_save_freq', type=int, default=10,
                        help='Save checkpoint frequency')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_points', type=int, default=4000,
                        help='Number of points for visualization')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only visualize results from existing checkpoints (skip training)')
    parser.add_argument('--vis_epoch_step', type=int, default=20,
                        help='Epoch step for visualization progress (default: 20)')
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

    checkpoint_prefix = f'{args.output_dir}/dmd2_{args.dataset}_steps{args.student_steps}'

    if args.visualize_only:
        # Visualize existing checkpoints without training
        print("\nGenerating visualization from existing checkpoints...")
        initial_points = source_generator.generate(args.num_points)
        target_points = target_generator.generate(args.num_points).numpy()

        visualize_dmd2_progress(
            n_epochs=args.epochs,
            initial_points=initial_points,
            teacher_model=teacher_model,
            target_points=target_points,
            student_steps_list=[args.student_steps, 2],
            teacher_steps=100,
            model_kwargs={'input_dim': 2, 'h_dim': args.h_dim},
            checkpoint_prefix=checkpoint_prefix,
            epoch_step=args.vis_epoch_step,
            device=device,
            suptitle=f"DMD2 on {args.dataset.capitalize()} (Student trained with {args.student_steps} step{'s' if args.student_steps > 1 else ''})"
        )
        plt.savefig(f'./imgs/dmd2_progress_{args.dataset}_steps{args.student_steps}.png', bbox_inches='tight', dpi=150)
        print(f"Saved visualization to ./imgs/dmd2_progress_{args.dataset}_steps{args.student_steps}.png")
        plt.show()
    else:
        # Initialize proxy model from teacher (learns fake distribution)
        proxy_model = VelocityNet(input_dim=2, h_dim=args.h_dim).to(device)
        proxy_model.load_state_dict(teacher_model.state_dict())
        print("Initialized proxy model from teacher checkpoint")

        # Initialize student model from teacher
        student_model = VelocityNet(input_dim=2, h_dim=args.h_dim).to(device)
        student_model.load_state_dict(teacher_model.state_dict())
        print("Initialized student model from teacher checkpoint")

        print(f"\nStarting DMD2 training:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Student steps: {args.student_steps}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Proxy update ratio: {args.proxy_update_ratio}")

        start_time = time.time()

        student_model, proxy_model = train_dmd2(
            teacher_model=teacher_model,
            proxy_model=proxy_model,
            student_model=student_model,
            source_generator=source_generator,
            n_epochs=args.epochs,
            lr_student=args.lr_student,
            lr_proxy=args.lr_proxy,
            batch_size=args.batch_size,
            batches_per_epoch=50,
            student_steps=args.student_steps,
            proxy_update_ratio=args.proxy_update_ratio,
            epoch_save_freq=args.epoch_save_freq,
            checkpoint_prefix=checkpoint_prefix,
            device=device
        )

        elapsed_time = time.time() - start_time
        print(f"\nTraining took {elapsed_time:.2f} seconds")

        # Visualize results
        print("\nGenerating visualization...")
        visualize_dmd2_results(
            student_model=student_model,
            teacher_model=teacher_model,
            source_generator=source_generator,
            target_generator=target_generator,
            num_points=args.num_points,
            student_steps=args.student_steps,
            teacher_steps=100,
            device=device,
            save_path=f'./imgs/dmd2_{args.dataset}_steps{args.student_steps}.png'
        )