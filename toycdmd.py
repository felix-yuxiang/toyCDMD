"""
CDMD (Consistency Distribution Matching Distillation) for 2D Toy Datasets
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
    student_steps: int,
    t: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute CDMD distribution matching loss.

    The key idea:
    - Student generates samples from noise in few steps
    - Teacher model predicts velocities at interpolated points
    - Loss encourages student to match teacher's distribution

    Args:
        noise: Input noise samples (batch_size, dim)
        student_model: Student model (generates in few steps)
        teacher_model: Pre-trained teacher (frozen)
        student_steps: Number of steps for student generation
        t: Random timesteps for interpolation
        device: Device to use

    Returns:
        Distribution matching loss
    """
    batch_size = noise.shape[0]

    # Student generates samples from noise (with gradients)
    generated_samples = forward_euler_integration(
        noise, student_model, t_start=0.99, t_end=0.0,
        num_steps=student_steps, device=device
    )

    # print the norm average of genrated_samples
    print("Generated samples norm average:", torch.norm(generated_samples, dim=1).mean().item())

    # Create interpolated samples: x_t = (1-t) * x_generated + t * noise
    # This follows flow matching interpolation convention
    # noise = torch.randn_like(noise).to(device) 
    # Note t2 > t1
    x_t1 = (1 - t) * generated_samples + t * noise


    gap_min = 0.01
    gap_max = 0.30
    gap = torch.rand(batch_size, 1, device=device) * (gap_max - gap_min) + gap_min

    t2 = t + gap
    x_t2 = (1 - t2) * generated_samples + t2 * noise

    # Get velocity predictions from teacher (frozen)
    with torch.no_grad():
        # Step 1: Get teacher velocity at t2
        teacher_velocity_t2 = teacher_model(x_t2, t2)

        # Step 2: Estimate x0 using teacher's prediction at t2
        # From x_t = (1-t)*x_0 + t*noise and v = noise - x_0, we get x_0 = x_t - t*v
        # teacher_x0 = x_t2 - t2 * teacher_velocity_t2

        # Step 3: Re-interpolate to time t using teacher's x0 estimate
        # xt_next_hat = (1 - t) * teacher_x0 + t * noise


        xt_next_hat = x_t2 - (t2 - t) * teacher_velocity_t2

        # Step 4: Get velocities at time t for both paths
        # - teacher_velocity: velocity at the "corrected" point (using teacher's x0)
        # - student_velocity: velocity at the student's interpolated point
        teacher_velocity = teacher_model(xt_next_hat, t)

        ##### ??????
        # student_velocity = teacher_model(x_t1, t)
        student_velocity = noise - generated_samples

        # Distribution matching loss: difference between the two velocity predictions
        p_diff = (teacher_velocity - student_velocity) * t
        weight_factor = torch.abs(generated_samples - x_t2 + t2 * teacher_velocity_t2).mean(dim=1, keepdim=True)
        # p_diff = (teacher_velocity - student_velocity) 
        # weight_factor = torch.abs(teacher_velocity).mean(dim=1, keepdim=True).clamp(min=1e-4)
        grad = p_diff / weight_factor
        grad = torch.nan_to_num(grad)
        # grad = p_diff

    # The loss encourages generated samples to move towards teacher distribution
    loss_dm = 0.5 * F.mse_loss(generated_samples, (generated_samples - grad).detach())

    return loss_dm, generated_samples




def train_cdmd(
    teacher_model: nn.Module,
    student_model: nn.Module,
    source_generator,
    n_epochs: int = 100,
    lr_student: float = 1e-4,
    batch_size: int = 2048,
    batches_per_epoch: int = 50,
    student_steps: int = 1,
    epoch_save_freq: int = 10,
    checkpoint_prefix: str = 'cdmd_model',
    device: str = 'cpu'
):
    """
    CDMD training loop

    Args:
        teacher_model: Pre-trained teacher (frozen)
        student_model: Student model to train
        source_generator: Noise generator
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

    optimizer_student = optim.Adam(student_model.parameters(), lr=lr_student)

    step = 0
    for epoch in range(n_epochs):
        student_model.train()

        total_loss_dm = 0.0

        for batch_idx in range(batches_per_epoch):
            noise = source_generator.generate(batch_size).to(device)
            start, end = 0.0, 0.69
            t = torch.rand(batch_size, 1, device=device) * (end - start) + start

            # Update student model
            optimizer_student.zero_grad()
            loss_dm, _ = compute_distribution_matching_loss(
                noise, student_model, teacher_model,
                student_steps, t, device
            )
            loss_dm.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer_student.step()

            total_loss_dm += loss_dm.item()
            step += 1

        avg_loss_dm = total_loss_dm / batches_per_epoch
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss DM: {avg_loss_dm:.6f}")

        if epoch % epoch_save_freq == 0:
            checkpoint_path = f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_model.state_dict(),
                'optimizer_student_state_dict': optimizer_student.state_dict(),
                'loss_dm': avg_loss_dm,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    checkpoint_path = f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'student_state_dict': student_model.state_dict(),
        'optimizer_student_state_dict': optimizer_student.state_dict(),
        'loss_dm': avg_loss_dm,
    }, checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")

    return student_model


def visualize_cdmd_results(
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
    """Visualize CDMD results comparing student and teacher"""
    student_model.eval()
    teacher_model.eval()

    noise = source_generator.generate(num_points).to(device)
    target_points = target_generator.generate(num_points).cpu().numpy()

    # Student generation (few steps)
    student_samples = forward_euler_integration_no_grad(
        noise, student_model, t_start=1.0, t_end=0.0,
        num_steps=student_steps, device=device
    ).cpu().numpy()

    ### print the student samples
    print("Student samples:")
    print(student_samples)
    print("Target points:")
    print(target_points)

    # Teacher generation (many steps)
    teacher_samples = forward_euler_integration_no_grad(
        noise, teacher_model, t_start=1.0, t_end=0.0,
        num_steps=teacher_steps, device=device
    ).cpu().numpy()

    # Compute teacher_x0 estimate (single-step x0 prediction from noise)
    # Using the same logic as in compute_distribution_matching_loss
    with torch.no_grad():
        t = torch.ones(num_points, 1, device=device) * 0.99  # Start from t close to 1
        x_t = noise  # At t=1, x_t is approximately noise
        teacher_velocity = teacher_model(x_t, t)
        # x_0 = x_t - t * v (from flow matching: x_t = (1-t)*x_0 + t*noise, v = noise - x_0)
        teacher_x0 = (x_t - t * teacher_velocity).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Target distribution
    axes[0].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.3, color='blue')
    axes[0].set_title('Target Distribution', fontsize=14)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Teacher generation (multi-step)
    axes[1].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.1, color='blue', label='Target')
    axes[1].scatter(teacher_samples[:, 0], teacher_samples[:, 1], s=5, alpha=0.3, color='green', label='Teacher')
    axes[1].set_title(f'Teacher ({teacher_steps} steps)', fontsize=14)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Teacher x0 estimate (single-step prediction)
    axes[2].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.1, color='blue', label='Target')
    axes[2].scatter(teacher_x0[:, 0], teacher_x0[:, 1], s=5, alpha=0.3, color='orange', label='Teacher x0')
    axes[2].set_title('Teacher x0 (1-step estimate)', fontsize=14)
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Student generation
    axes[3].scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.1, color='blue', label='Target')
    axes[3].scatter(student_samples[:, 0], student_samples[:, 1], s=5, alpha=0.3, color='red', label='Student')
    axes[3].set_title(f'Student ({student_steps} step{"s" if student_steps > 1 else ""})', fontsize=14)
    axes[3].set_xlim(-3, 3)
    axes[3].set_ylim(-3, 3)
    axes[3].set_aspect('equal')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

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


def load_cdmd_student_model(checkpoint_path: str, input_dim: int = 2, h_dim: int = 128, device: str = 'cpu'):
    """Load trained CDMD student model from checkpoint"""
    model = VelocityNet(input_dim=input_dim, h_dim=h_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['student_state_dict'])
    model.eval()
    print(f"Loaded student model from {checkpoint_path}")
    return model


def visualize_cdmd_progress(
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
    suptitle: str = "CDMD Training Progress"
):
    """
    Visualizes CDMD student model results across training epochs.

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
    n_cols = len(student_steps_list) + 1  # +1 for teacher
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Generate teacher samples once (reference)
    teacher_samples = forward_euler_integration_no_grad(
        initial_points, teacher_model, t_start=1.0, t_end=0.0,
        num_steps=teacher_steps, device=device
    ).cpu().numpy()

    for i, epoch in enumerate(steps):
        model_name = f'{checkpoint_prefix}_epoch_{epoch}.pt'
        print(f"Loading model {model_name}")

        # Load student model
        student_model = VelocityNet(**model_kwargs).to(device)
        checkpoint = torch.load(model_name, map_location=device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        student_model.eval()

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

        # Student with different step counts
        for j, num_steps in enumerate(student_steps_list):
            student_samples = forward_euler_integration_no_grad(
                initial_points, student_model, t_start=1.0, t_end=0.0,
                num_steps=num_steps, device=device
            ).cpu().numpy()

            ax = axes[i, j + 1]
            ax.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
            ax.scatter(student_samples[:, 0], student_samples[:, 1], s=5, alpha=0.3, color='yellow', label='Student')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'Student ({num_steps} step{"s" if num_steps > 1 else ""})', fontsize=16)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=20, y=1.02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CDMD training for 2D toy datasets')
    parser.add_argument('--teacher_checkpoint', type=str,
                        default='./models/flow_model_spiral_epoch_100.pt',
                        help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--dataset', type=str, default='spiral',
                        choices=['checkerboard', 'crescent', 'spiral'],
                        help='Target dataset (should match teacher training)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of CDMD training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr_student', type=float, default=1e-4,
                        help='Learning rate for student model')
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--student_steps', type=int, default=1,
                        help='Number of integration steps for student (1 for single-step)')
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

    checkpoint_prefix = f'{args.output_dir}/cdmd_{args.dataset}_steps{args.student_steps}'

    if args.visualize_only:
        # Visualize existing checkpoints without training
        print("\nGenerating visualization from existing checkpoints...")
        initial_points = source_generator.generate(args.num_points)
        target_points = target_generator.generate(args.num_points).numpy()

        visualize_cdmd_progress(
            n_epochs=args.epochs,
            initial_points=initial_points,
            teacher_model=teacher_model,
            target_points=target_points,
            student_steps_list=[args.student_steps],
            teacher_steps=100,
            model_kwargs={'input_dim': 2, 'h_dim': args.h_dim},
            checkpoint_prefix=checkpoint_prefix,
            epoch_step=args.vis_epoch_step,
            device=device,
            suptitle=f"CDMD on {args.dataset.capitalize()} (Student trained with {args.student_steps} step{'s' if args.student_steps > 1 else ''})"
        )
        plt.savefig(f'./imgs/cdmd_progress_{args.dataset}_steps{args.student_steps}.png', bbox_inches='tight', dpi=150)
        print(f"Saved visualization to ./imgs/cdmd_progress_{args.dataset}_steps{args.student_steps}.png")
        plt.show()
    else:
        # Initialize student model from teacher
        student_model = VelocityNet(input_dim=2, h_dim=args.h_dim).to(device)
        student_model.load_state_dict(teacher_model.state_dict())
        print("Initialized student model from teacher checkpoint")

        print(f"\nStarting CDMD training:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Student steps: {args.student_steps}")
        print(f"  Epochs: {args.epochs}")

        start_time = time.time()

        student_model = train_cdmd(
            teacher_model=teacher_model,
            student_model=student_model,
            source_generator=source_generator,
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
        visualize_cdmd_results(
            student_model=student_model,
            teacher_model=teacher_model,
            source_generator=source_generator,
            target_generator=target_generator,
            num_points=args.num_points,
            student_steps=args.student_steps,
            teacher_steps=100,
            device=device,
            save_path=f'./imgs/cdmd_{args.dataset}_steps{args.student_steps}.png'
        )