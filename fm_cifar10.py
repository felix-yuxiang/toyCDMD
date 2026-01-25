import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from unet import SongUNet


class FlowMatching:
    """Flow Matching model using UNet backbone."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def get_velocity(self, x0, x1):
        """Compute velocity from target (x0) to source (x1).

        In flow matching, we interpolate: x_t = (1-t) * x0 + t * x1
        where x0 is data (t=0) and x1 is noise (t=1).
        Velocity v = x1 - x0 (takes data to noise).
        """
        return x1 - x0

    def get_interpolated(self, x0, x1, t):
        """Get interpolated sample at time t.

        x_t = (1-t) * x0 + t * x1
        """
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_loss(self, x0, x1, t):
        """Compute flow matching loss.

        Args:
            x0: Data samples (batch_size, C, H, W)
            x1: Noise samples (batch_size, C, H, W)
            t: Time values (batch_size,)

        Returns:
            MSE loss between predicted and true velocity
        """
        # Get interpolated samples
        x_t = self.get_interpolated(x0, x1, t)

        # True velocity
        velocity_true = self.get_velocity(x0, x1)

        # Predicted velocity from model
        # UNet expects (x, t, s) where s is step size - for flow matching we use t as step
        t_input = t.view(-1)
        s_input = torch.zeros_like(t_input)  # Not used in standard flow matching
        velocity_pred = self.model(x_t, t_input, s_input)

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_true)
        return loss


def forward_euler_integration(
    model,
    initial_noise: torch.Tensor,
    num_steps: int = 100,
    device='cuda'
) -> torch.Tensor:
    """Integrate from noise (t=1) to data (t=0) using forward Euler.

    Args:
        model: UNet velocity model
        initial_noise: Starting noise tensor (batch_size, C, H, W)
        num_steps: Number of integration steps
        device: Torch device

    Returns:
        Generated samples at t=0
    """
    dt = 1.0 / num_steps
    x = initial_noise.clone().to(device)

    for step in range(num_steps):
        t = 1.0 - step * dt  # Start at t=1, go to t=0
        t_tensor = torch.full((x.shape[0],), t, device=device)
        s_tensor = torch.zeros_like(t_tensor)

        with torch.no_grad():
            velocity = model(x, t_tensor, s_tensor)

        # Move towards data: x -= velocity * dt (since velocity = noise - data)
        x = x - velocity * dt

    return x


def train_flow_matching(
    model,
    train_loader,
    n_epochs=100,
    lr=1e-4,
    device='cuda',
    checkpoint_dir='./checkpoints',
    save_freq=10,
    log_freq=100
):
    """Train flow matching model on CIFAR-10.

    Args:
        model: UNet model
        train_loader: DataLoader for CIFAR-10
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Torch device
        checkpoint_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N epochs
        log_freq: Log every N batches

    Returns:
        Trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    fm = FlowMatching(model, device)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # Data samples (x0)
            x0 = data.to(device)
            batch_size = x0.shape[0]

            # Noise samples (x1)
            x1 = torch.randn_like(x0)

            # Random time values
            t = torch.rand(batch_size, device=device)

            # Compute loss
            loss = fm.compute_loss(x0, x1, t)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_freq == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'fm_cifar10_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    return model


@torch.no_grad()
def generate_samples(model, num_samples=64, num_steps=100, device='cuda'):
    """Generate samples using the trained model.

    Args:
        model: Trained UNet model
        num_samples: Number of samples to generate
        num_steps: Number of integration steps
        device: Torch device

    Returns:
        Generated samples (num_samples, 3, 32, 32)
    """
    model.eval()

    # Start from noise
    noise = torch.randn(num_samples, 3, 32, 32, device=device)

    # Integrate from t=1 to t=0
    samples = forward_euler_integration(model, noise, num_steps=num_steps, device=device)

    # Clamp to valid range
    samples = torch.clamp(samples, -1, 1)

    return samples


def visualize_samples(samples, save_path=None, nrow=8):
    """Visualize generated samples.

    Args:
        samples: Generated samples tensor (N, 3, 32, 32)
        save_path: Path to save the figure
        nrow: Number of images per row
    """
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Create grid
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=nrow, padding=2)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title('Generated CIFAR-10 Samples')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.close()


def get_cifar10_dataloader(batch_size=128, num_workers=4, data_dir='./data'):
    """Get CIFAR-10 data loader.

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to store/load CIFAR-10 data

    Returns:
        DataLoader for CIFAR-10 training set
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    return train_loader


def create_unet_model(device='cuda'):
    """Create UNet model for CIFAR-10 (32x32 images).

    Returns:
        SongUNet model configured for CIFAR-10
    """
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,  # Unconditional
        augment_dim=0,
        model_channels=128,
        channel_mult=[1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0,
        embedding_type='positional',
        channel_mult_noise=1,
        encoder_type='standard',
        decoder_type='standard',
        resample_filter=[1, 1],
    ).to(device)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Flow Matching on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/fm_cifar10',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log every N batches (default: 100)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate samples from a trained model')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate (default: 64)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of integration steps for generation (default: 100)')
    parser.add_argument('--output_image', type=str, default='./imgs/fm_cifar10_samples.png',
                        help='Output path for generated samples image')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_unet_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    if args.generate_only:
        # Generate samples only
        if args.resume is None:
            raise ValueError("Must specify --resume with a trained model for generation")

        print(f"Generating {args.num_samples} samples with {args.num_steps} integration steps...")
        samples = generate_samples(model, args.num_samples, args.num_steps, device)

        os.makedirs(os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else '.', exist_ok=True)
        visualize_samples(samples, args.output_image)
    else:
        # Training mode
        print("Loading CIFAR-10 dataset...")
        train_loader = get_cifar10_dataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir
        )
        print(f"Training set size: {len(train_loader.dataset)}")
        print(f"Batches per epoch: {len(train_loader)}")

        print("\nStarting training...")
        start_time = time.time()

        trained_model = train_flow_matching(
            model=model,
            train_loader=train_loader,
            n_epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            save_freq=args.save_freq,
            log_freq=args.log_freq
        )

        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/3600:.2f} hours")

        # Generate samples after training
        print("\nGenerating samples...")
        samples = generate_samples(trained_model, args.num_samples, args.num_steps, device)

        os.makedirs(os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else '.', exist_ok=True)
        visualize_samples(samples, args.output_image)
