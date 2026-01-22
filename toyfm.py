import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.optim as optim


class GaussianGenerator:
    def __init__(self, n_dims=2, noise_std=1.0):
        self.n_dims = n_dims
        self.noise_std = noise_std

    def generate(self, num_points):
        return torch.randn(num_points, self.n_dims) * self.noise_std

class CrescentGenerator:
    def __init__(self, R=1.0, r=0.6, d=0.5):
        self.R = R  # Outer radius
        self.r = r  # Inner circle radius
        self.d = d  # Offset of inner circle

    def generate(self, num_points):
        # Calculate the area ratio to estimate required samples
        outer_area = np.pi * self.R**2
        inner_area = np.pi * self.r**2
        crescent_area = outer_area - inner_area

        # Estimate required samples with 20% buffer
        n_samples = int(num_points * (outer_area / crescent_area) * 1.2)
        n_samples = max(n_samples, num_points)  # Ensure we generate at least num_points

        # Generate points in the outer circle
        theta = 2 * np.pi * torch.rand(n_samples)
        radius = self.R * torch.sqrt(torch.rand(n_samples))

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        # Filter points that are outside the inner circle
        mask = (x - self.d)**2 + y**2 > self.r**2
        points = torch.stack((x[mask], y[mask]), dim=1)

        # If we didn't get enough points, recursively generate more
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



def visualize_denoising_progress(
    n_epochs,
    initial_points,
    num_steps_multi,
    model_class,
    model_kwargs,
    target_points,
    integration_fn,
    checkpoint_prefix='flow_model',
    epoch_step=10,
    device='cpu',
    suptitle="Model Performance Over Time"
):
    """
    Visualizes denoising results across training epochs using two schemes:
    multi-step and single-step denoising.

    Args:
        n_epochs: Total number of training epochs.
        initial_points: Noisy input points (torch.Tensor).
        num_steps_multi: Integration steps for multi-step denoising.
        model_class: fm 
        model_kwargs: Dict of kwargs to initialize model (e.g., {'input_dim': 2, 'h_dim': 128}).
        target_points: Target distribution (numpy array).
        integration_fn: Function that performs forward integration.
        checkpoint_prefix: Prefix of checkpoint files.
        epoch_step: Interval at which to visualize checkpoints.
        device: Torch device.
        suptitle: Title for the entire figure.
    """
    steps = list(range(0, n_epochs + 1, epoch_step))
    n_rows = len(steps)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    for i, epoch in enumerate(steps):
        model_name = f'{checkpoint_prefix}_epoch_{epoch}.pt'
        print(f"Loading model {model_name}")

        # Instantiate and load model
        plot_model = model_class(**model_kwargs).to(device)
        checkpoint = torch.load(model_name, map_location=device)
        plot_model.load_state_dict(checkpoint['model_state_dict'])
        plot_model.eval()

        # Multi-step denoising
        denoised_multi = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0, #we start at t = 1, corresponding to noise
            t_end=0.0, #and end at t = 0, corresponding to data
            num_steps=num_steps_multi,
            save_trajectory=False,
            device = device
        ).cpu().numpy()


        denoised_multi_half = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0, #we start at t = 1, corresponding to noise
            t_end=0.0, #and end at t = 0, corresponding to data
            num_steps=num_steps_multi//2,
            save_trajectory=False,
            device = device
        ).cpu().numpy()

        denoised_multi_sm = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0, #we start at t = 1, corresponding to noise
            t_end=0.0, #and end at t = 0, corresponding to data
            num_steps=num_steps_multi//10,
            save_trajectory=False,
            device = device
        ).cpu().numpy()

        # Single-step denoising
        denoised_single = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0,
            t_end=0.0,
            num_steps=1,
            save_trajectory=False,
            device = device
        ).cpu().numpy()

        # Left: multi-step
        ax_left = axes[i, 0]
        ax_left.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_left.scatter(denoised_multi[:, 0], denoised_multi[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_left.set_xlim(-3, 3)
        ax_left.set_ylim(-3, 3)
        ax_left.set_aspect('equal')
        if i == 0:
            ax_left.set_title(f'Denoising Steps = {num_steps_multi}', fontsize=20)
        ax_left.set_ylabel(f"Epoch {epoch}", fontsize=16)
        ax_left.grid(True, alpha=0.3)

        ax_left = axes[i, 1]
        ax_left.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_left.scatter(denoised_multi_half[:, 0], denoised_multi_half[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_left.set_xlim(-3, 3)
        ax_left.set_ylim(-3, 3)
        ax_left.set_aspect('equal')
        if i == 0:
            ax_left.set_title(f'Denoising Steps = {num_steps_multi//2}', fontsize=20)
        ax_left.set_ylabel(f"Epoch {epoch}", fontsize=16)
        ax_left.grid(True, alpha=0.3)

        ax_left = axes[i, 2]
        ax_left.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_left.scatter(denoised_multi_sm[:, 0], denoised_multi_sm[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_left.set_xlim(-3, 3)
        ax_left.set_ylim(-3, 3)
        ax_left.set_aspect('equal')
        if i == 0:
            ax_left.set_title(f'Denoising Steps = {num_steps_multi//10}', fontsize=20)
        ax_left.set_ylabel(f"Epoch {epoch}", fontsize=16)
        ax_left.grid(True, alpha=0.3)

        # Right: single-step
        ax_right = axes[i, 3]
        ax_right.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_right.scatter(denoised_single[:, 0], denoised_single[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_right.set_xlim(-3, 3)
        ax_right.set_ylim(-3, 3)
        ax_right.set_aspect('equal')
        if i == 0:
            ax_right.set_title('Denoising Steps = 1', fontsize=20)
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=25, y=1.02)

def forward_euler_integration_model(
    initial_points: torch.Tensor,
    model: nn.Module,
    t_start: float = 1.0,
    t_end: float = 0.0,
    num_steps: int = 100,
    save_trajectory: bool = True,
    device = 'cpu'
) -> torch.Tensor:

    dt = (t_start - t_end) / num_steps
    trajectory = [initial_points.clone()] if save_trajectory else None
    current_points = initial_points.clone().to(device)

    for step in range(0, num_steps):
        current_time = t_start - step*dt
        t_tensor = torch.full((len(current_points), 1), current_time,
                            device=current_points.device)

        with torch.no_grad():
            velocity = model(current_points, t_tensor)

        current_points += -velocity * dt # velocity takes us from targets to source

        if save_trajectory:
            trajectory.append(current_points.clone())

    return torch.stack(trajectory) if save_trajectory else current_points


class VelocityNet(nn.Module):
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.fc_in  = nn.Linear(input_dim + 1, h_dim)
        self.fc2    = nn.Linear(h_dim, h_dim)
        self.fc3    = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)

def train_model(model, source_data_function, target_data_function, n_epochs=100, lr=0.003, batch_size=2048, batches_per_epoch=10, epoch_save_freq = 10, checkpoint_prefix='flow_model'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            optimizer.zero_grad()

            # obtain points
            source_samples = source_data_function(batch_size).to(device)
            target_samples = target_data_function(batch_size).to(device)

            t = torch.rand(source_samples.size(0), 1).to(device)  # random times for training
            interpolated_samples = (1 - t) * target_samples + t * source_samples
            velocity = source_samples - target_samples # velocity takes targets to sources

            velocity_prediction = model(interpolated_samples, t)
            loss = loss_fn(velocity_prediction, velocity)

            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / batches_per_epoch
        print(f"Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_loss:.4f}")

        if epoch % epoch_save_freq == 0:
            # Save model checkpoint
            checkpoint_path = f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
    # Always save final model at the end
    checkpoint_path = f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flow matching model on 2D toy datasets')
    parser.add_argument('--dataset', type=str, default='checkerboard',
                        choices=['checkerboard', 'crescent', 'spiral'],
                        help='Target dataset to use (default: checkerboard)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs ')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training (default: 4096)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate (default: 0.003)')
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Hidden dimension of velocity network (default: 128)')
    parser.add_argument('--num_points', type=int, default=4000,
                        help='Number of points for visualization (default: 4000)')
    parser.add_argument('--epoch_save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of integration steps for visualization (default: 100)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for checkpoints (default: current directory)')
    args = parser.parse_args()

    # Source generator (Gaussian noise)
    source_generator = GaussianGenerator(n_dims=2, noise_std=1.0)

    # Select target generator based on argument
    if args.dataset == 'checkerboard':
        target_generator = CheckerboardGenerator(grid_size=4, scale=1.5)
    elif args.dataset == 'crescent':
        target_generator = CrescentGenerator(R=1.0, r=0.6, d=0.5)
    elif args.dataset == 'spiral':
        target_generator = SpiralGenerator(noise_std=0.05, n_turns=3, radius_scale=2)

    print(f"Training flow matching model on {args.dataset} dataset")

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VelocityNet(input_dim=2, h_dim=args.h_dim).to(device)
    checkpoint_prefix = f'{args.output_dir}/flow_model_{args.dataset}'

    # Train the model
    start_time = time.time()

    trained_model = train_model(
        model=model,
        source_data_function=source_generator.generate,
        target_data_function=target_generator.generate,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        batches_per_epoch=50,
        epoch_save_freq=args.epoch_save_freq,
        checkpoint_prefix=checkpoint_prefix
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")

    # Visualize denoising progress
    print("Generating visualization...")
    initial_points = source_generator.generate(args.num_points)
    target_points = target_generator.generate(args.num_points).numpy()

    visualize_denoising_progress(
        n_epochs=args.epochs,
        initial_points=initial_points,
        num_steps_multi=args.num_steps,
        model_class=VelocityNet,
        model_kwargs={'input_dim': 2, 'h_dim': args.h_dim},
        target_points=target_points,
        integration_fn=forward_euler_integration_model,
        checkpoint_prefix=checkpoint_prefix,
        epoch_step=20,
        device=device,
        suptitle=f"Flow Matching on {args.dataset.capitalize()} Dataset"
    )
    plt.savefig(f'./imgs/fm_denoising_progress_{args.dataset}.png', bbox_inches='tight')