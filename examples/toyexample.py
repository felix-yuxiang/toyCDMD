import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from smalldiffusion import (
    TimeInputMLP, ScheduleLogLinear, training_loop, training_loop_and_eval, samples,
    DatasaurusDozen, Swissroll, Checkerboard, PredX0, Scaled, ScheduleCosine
)



def plot_batch(batch):
    batch = batch.cpu().numpy()
    # plt.scatter(batch[:,0], batch[:,1], marker='.')

    plt.hist2d(batch[:, 0], batch[:, 1], bins=100, cmap='viridis', density=True, range=[[-1, 1], [-1, 1]])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    # # Remove axis ticks and labels for the clean look
    # plt.axis('off')
    
    # # Ensure the plot isn't stretched
    # plt.gca().set_aspect('equal')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

dataset_name = 'swiss'
# dataset = DatasaurusDozen(csv_file='../datasets/DatasaurusDozen.tsv', dataset='dino')
# Or use the SwissRoll dataset
# dataset = Swissroll(3, 12, 10000)
dataset = Swissroll(3, 12, 10000)
# dataset = Checkerboard(N=2500)
loader = DataLoader(dataset, batch_size=2500)
plot_batch(next(iter(loader)))
plt.savefig(f'../outputs/toyexample_dataset_{dataset_name}.png')
plt.clf()


### pred x0
model = TimeInputMLP(hidden_dims=(16,64,128,128,128,128,64,16))
print(model)

schedule = ScheduleLogLinear(N=200, sigma_min=0.02, sigma_max=10)
# schedule = ScheduleCosine(N=200)
trainer = training_loop_and_eval(loader, model, schedule, epochs=15000, lr=1e-3, plot_batch=plot_batch, dataset_name=dataset_name)
losses = [ns.loss.item() for ns in trainer]


### save the model
os.makedirs('../models', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)
torch.save(model.state_dict(), f'../models/toyexample_model_{dataset_name}.pth')

plt.clf()
plt.plot(moving_average(losses, 100))
plt.savefig(f'../outputs/toyexample_training_loss_{dataset_name}.png')
plt.clf()

# For DDPM sampling, change to gam=1, mu=0.5
# For DDIM sampling, change to gam=1, mu=0
schedule = ScheduleLogLinear(N=200, sigma_min=0.02, sigma_max=10)
*xts, x0 = samples(model, schedule.sample_sigmas(20), batchsize=1500, gam=2, mu=0)
plot_batch(x0)
plt.savefig(f'../outputs/toyexample_generated_samples_{dataset_name}.png')



