import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Image as IPImage
from wiring_efficiency_utils import *
from neuralsheet import *
from map_plotting import *
import os


def collect_dim_stats():
    
    # Example usage
    crop_size = 60 # Crop size (NxN)
    batch_size = 64  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = 'cuda'  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    std_rfs = 5
    beta = 1 - 5e-5
    loss_beta = 3e-3

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 11
    trialvar = np.sqrt(np.linspace(0.4**2, 2.4**2, trials))
    #trialvar = [0.5, 0.625, 0.875, 1]
    #sizesvar = np.round(np.sqrt(np.linspace(400, 2500, 3))).astype(int)
    sizesvar = [20, 30, 50]
    sizes = len(sizesvar)
    trials = len(trialvar)
    #noise = np.linspace(0., 0.6, trials)
    print(trialvar, sizesvar)
    epochs = 3
    reco_tracker = torch.zeros((sizes, trials, len(dataloader)))
    se_tracker = torch.zeros((sizes, trials))
    map_tracker = torch.zeros((sizes, trials, sizesvar[-1], sizesvar[-1]))
    spectrum_tracker = torch.zeros((sizes, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((sizes, trials))

    tuning_curve_tracker = torch.zeros((sizes, trials, 101))
    n_samples = 3
    comp_tracker = torch.zeros((sizes, trials, n_samples*sizesvar[-1], n_samples*sizesvar[-1]))
    se_pca_tracker = torch.zeros((sizes, trials))

    code_tracker = []

    for s in range(sizes):
        for t in range(trials):

            model = NeuralSheet(sizesvar[s], crop_size, trialvar[t], std_rfs, device=device).to(device)
            lr = 1e-3
            rf_grids = get_grids(crop_size, crop_size, model.rf_size, sizesvar[s], device=device)

            network = init_nn(sizesvar[s], crop_size)
            avg_loss = 0
            code_tracker = []

            for e in range(epochs):

                batch_progress = tqdm(dataloader, leave=False)
                for b_idx, batch in enumerate(batch_progress):

                    batch_responses = []
                    batch_inputs = []
                    batch = batch.to('cuda')  # Transfer the entire batch to GPU

                    for image in batch:

                        image = image[0:1][None].flip(1)

                        if image.mean()>0.15:

                            limit = 1e-4
                            lr *= beta
                            lr = lr if lr>limit else limit

                            model.hebbian_lr = lr
                            model.homeo_lr = lr

                            model(image, rf_grids, noise_lvl=0)
                            model.hebbian_step()

                            batch_responses.append(model.current_response.clone())
                            batch_inputs.append(model.current_input.clone())
                            code_tracker.append(model.current_response.clone())

                    batch_responses = torch.cat(batch_responses, dim=0)
                    batch_inputs = torch.cat(batch_inputs, dim=0)

                    reco_input = network['activ'](network['model'](batch_responses))

                    targets = batch_inputs
                    loss, loss_std = nn_loss(network, targets, reco_input)

                    sim = cosim(targets.detach().cpu(), reco_input.detach().cpu(), True)
                    sim = (sim - 0.6) / 0.4
                    reco_tracker[s, t, b_idx] = sim

                    #se_mean = get_spectral_entropy(targets)       
                    avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim

                    network['optim'].zero_grad()
                    loss.backward()
                    network['optim'].step()

                    if b_idx%50==0:
                        ori_map, phase_map, mean_tc = get_orientations(model.afferent_weights, gabor_size=model.rf_size)

                    mean_activation = model.l4_mean_activations.mean()
                    mean_std = model.l4_mean_activations.std() / model.homeo_target
                    batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}, RF:{:.4f}'.format(
                        mean_activation, 
                        mean_std, 
                        avg_loss,
                        lr,
                        mean_tc.max()
                    ))

                    #if len(code_tracker)>4000:  
                    #    break


            ori_map, phase_map, mean_tc = get_orientations(model.afferent_weights, gabor_size=model.rf_size)
            ori_map = ori_map.view(sizesvar[s], sizesvar[s]).cpu()

            eff_dims, spectrum, peak = get_effective_dims(code_tracker)
            eff_dims_pca, samp_components = get_pca_dimensions(code_tracker, n_samples)

            tuning_curve_tracker[s, t] = mean_tc
            se_tracker[s, t] = eff_dims
            se_pca_tracker[s, t] = eff_dims_pca

            comp_size = n_samples * sizesvar[s]
            comp_tracker[s, t, :comp_size, :comp_size] = samp_components

            print('Reco-SE / SEPCA / RF sharp so far: ', avg_loss, eff_dims, eff_dims_pca, mean_tc.max())

            map_tracker[s, t,:sizesvar[s],:sizesvar[s]] = ori_map
            spectrum_tracker[s,t,:sizesvar[s],:sizesvar[s]] = spectrum.cpu()
            peak_tracker[s,t] = peak.cpu()


    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'map_tracker' : map_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'comp_tracker': comp_tracker,
        'se_pca_tracker': se_pca_tracker,
        'tc_tracker' : tuning_curve_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar
    }

    torch.save(data, 'data.pt')
    
    os.system("shutdown -h 0")