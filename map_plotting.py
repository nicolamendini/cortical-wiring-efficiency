import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter

from wiring_efficiency_utils import *

def sample_and_plot(distribution, num_samples, sample_idx, full=False):

    M = distribution.shape[-1]

    # Convert distribution to PyTorch tensor and flatten for sampling
    dist_tensor = torch.tensor(distribution.flatten(), dtype=torch.float)
    
    # Sample S locations from the distribution
    indices = torch.multinomial(dist_tensor, num_samples, replacement=False)

    if full:
        indices = torch.where(dist_tensor>0)[0]
        num_samples = indices.shape[0]
    
    # Convert flat indices back to 2D indices
    y, x = np.unravel_index(indices.numpy(), (M, M))
    
    # Get the center coordinates
    center_x = sample_idx % M
    center_y = sample_idx // M
    
    # Visualization with Matplotlib
    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.imshow(distribution*0, cmap='binary', interpolation='nearest')
    
    # Add scatter to the sampled points with random scatter
    x_scatter = x + np.random.randn(num_samples) * 0.15  # Add random scatter to x coordinates
    y_scatter = y + np.random.randn(num_samples) * 0.15  # Add random scatter to y coordinates
    plt.scatter(x_scatter, y_scatter, color='black', s=10, alpha=0.5)  # Add transparency to the sampled points
    
    # Add scatter to the sampled points and draw lines from center to each point
    for i in range(len(x)):
        plt.plot([center_x, x[i]], [center_y, y[i]], color='black', linestyle='-', linewidth=0.5, alpha=0.)  # More transparent lines
    
    plt.scatter(center_x, center_y, color='black', s=100)  # Plot the center
    plt.title('Distribution with Sampled Locations')
    plt.axis('off')
    plt.subplot(1,2,2)
    # Load the image from local file
    image_path = "macaque_patches_v1.png"  # Change to your image filename
    image = Image.open(image_path)
    plt.title('Patchy Connectivity Macaque V1')
    # Display the rotated image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis for the image

    plt.show()


# Function to animate an array as a useful visualisation
def animate(array, n_frames, cmap=None, interval=300):
    
    fig = plt.figure(figsize=(6,6))
    global i
    i = -2
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    im = plt.imshow(array[0], animated=True, cmap=cmap)

    def updatefig(*args):
        global i
        if (i < n_frames - 1):  # ensure that we don't go out of bounds
            i += 1
        im.set_array(array[i])
        return im,

    anim = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=interval, repeat=True)
    plt.close(fig)  # Prevents the static plot from showing in the notebook
    return HTML(anim.to_jshtml())  # Directly display the animation


def show_map(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Inhibitory weights",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "Phase Map", "L4 Afferent", "L4 Histogram",
        "Reconstruction", "Positive Afferent", "Thresholds", "Mean activations"
    ]

    # Displaying the model's current input
    img = model.current_input[0, 0].detach().cpu()
    #c = model.rf_size // 2
    #img = img[c:-c,c:-c]
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap=cm.Greys)
    plt.title(titles[0])

    # Afferent weights of a random sample
    aff_weights = model.get_aff_weights()[random_sample, 0] #- model.afferent_weights[random_sample, 1]
    aff_weights[0,0] = 0
    plt.subplot(4, 4, 13)
    plt.imshow(aff_weights.detach().cpu())
    plt.title(titles[1])

    # Afferent weights of a random sample
    net_afferent = model.current_afferent[0,0].detach().cpu() - model.l4_thresholds[0,0].detach().cpu()
    net_afferent_bar = net_afferent + 0
    net_afferent_bar[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(net_afferent_bar)
    plt.title(titles[2])

    # Lateral correlations of the random sample
    plt.subplot(4, 4, 4)
    #plotvar = model.long_interactions[random_sample, 0]#* model.eye[random_sample, 0]
    plotvar = model.mid_range_inhibition[random_sample, 0]
    plotvar[0,0] = 0
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[3])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 4, 5)
    plotvar = model.l4_correlations[random_sample, 0]
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[4])

    # Model's current response
    plt.subplot(4, 4, 6)
    plt.imshow(model.current_response[0, 0].detach().cpu(), cmap=cm.Greys)
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0.1])
    plt.title(titles[6])

    # Generate and display orientation and phase maps
    weights = model.get_aff_weights().clone()
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map, phase_map, mean_tc = get_orientations(weights, gabor_size=model.rf_size)
    ori_map = ori_map.reshape(M, M).cpu()
    phase_map = phase_map.reshape(M, M).cpu()
    
    # Orientation map
    plt.subplot(4, 4, 8)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 9)
    hist_map = ori_map.flatten()
    plt.hist(hist_map, bins=15)
    plt.title(titles[8])

    # Phase map
    plt.subplot(4, 4, 10)
    plt.plot(mean_tc)
    plt.title(titles[9])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    detectors = get_detectors(model.rf_size, 1)
    plt.imshow(detectors[0,0].cpu())
    plt.title(titles[10])

    plt.subplot(4, 4, 12)
    plt.stairs(model.avg_l4_hist.int()[1:], torch.linspace(0.1,1,10), fill=True)
    plt.title(titles[11])

    reco_input = network['activ'](network['model'](model.current_response))[0,0].detach().cpu()
    # nn reconstruction
    plt.subplot(4, 4, 2)
    plt.imshow(reco_input, cmap=cm.Greys)
    plt.title(titles[12])

    # afferent with thresholds
    plt.subplot(4, 4, 14)
    plt.imshow(torch.relu(model.current_afferent - model.l4_thresholds)[0,0].cpu())
    plt.title(titles[13])

    # thresholds
    thresholds = model.l4_thresholds[0,0]
    #thresholds[0,0] = 0
    plt.subplot(4, 4, 15)
    plt.imshow(thresholds.cpu())
    plt.title(titles[14])

    # thresholds
    plt.subplot(4, 4, 16)
    mean_act = model.l4_mean_activations[0,0].cpu()
    mean_act[0,0] = 1
    plt.imshow(mean_act)
    plt.title(titles[15])

    print('Net Afferent Max: {:.3f}, Net Afferent Min: {:.3f}'. format(net_afferent.max(), net_afferent.min()))
    print('L4 Thresholds Max: {:.3f}, L4 Thresholds Min: {:.3f}'. format(model.l4_thresholds.max(), model.l4_thresholds.min()))
    print('Mean current response: {:.3f}'.format(model.current_response.mean()))
    print('L4 Strength: {:.3f} aff strength: {:.3f}'.format(model.l4_strength, model.aff_strength))
    loss = torch.mean((reco_input - img)**2)
    print('Reco loss: {:.3f}%'.format(loss))


    plt.show()
    
    
def make_compx_plots(data):
    
    #sns.set()
    
    #sizesvar = np.array(data['sizesvar'])
    sizesvar = [20,30,50]
    trialvar = np.array(data['trialvar'])
    fontsize = 20
    ticksize = 16
    
    # ----------------------- tuning curves
    tuning_curves = data['tc_tracker']
    tuning_curves -= tuning_curves.min(2, keepdim=True)[0]
    tuning_curves /= tuning_curves.max()
    
    orientations = np.linspace(0, np.pi, 5)
    
    for s in range(len(sizesvar)):
        
        plt.figure()
        plt.subplots_adjust(bottom=0.15, left=0.15)
        
        plt.xlabel('Orientation (rad)', fontsize=fontsize)
        plt.ylabel('Response (a.u.)', fontsize=fontsize)
        
        plt.xticks(orientations, labels=['0','π/4','π/2','3/4π','π'], fontsize=ticksize)
        plt.ylim(0, 1.05)
        plt.yticks([0, 0.5, 1], fontsize=ticksize)
                
        for i in range(len(trialvar)//2):
            
            curve_data = (tuning_curves[s, i*2])**2
            x = np.linspace(0, np.pi, tuning_curves.shape[-1])
            plt.plot(x, curve_data, label='σ={:.2f}'.format(trialvar[i*2]))
        
        plt.legend(fontsize=ticksize, frameon=False)
        plt.savefig(f'./fig1/tc{s}.svg')
        plt.close()
        
    # ----------------------- lambda
    peaks = data['peak_tracker']
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('Range of LE (σ)', fontsize=30)
    plt.ylabel('Char. wavelength (Λ)', fontsize=30)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    def linear_func(x, m, b):
        return m*x + b
    
    x = trialvar[None].repeat(len(sizesvar), axis=0).flatten()
    y = 1/peaks.flatten()

    lambda_popt, _ = curve_fit(linear_func, x, y)
    
    plt.plot(trialvar, linear_func(trialvar, lambda_popt[0], lambda_popt[1]), color='black', linewidth=1)

    for s in range(len(sizesvar)):
        ax.scatter(trialvar, 1/peaks[s], label='N='+str(sizesvar[s]**2), color='black', s=100)

    plt.savefig('./fig1/lambda.svg')
    plt.close()
        
        
    # ----------------------- accuracy curves
    
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.15)
    accuracy = data['reco_tracker'][:,:,-10000:].mean(2)
    
    plt.xlabel('Range of lateral excitation (σ)', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.ylim(0, 0.55)
    plt.yticks(fontsize=ticksize)
    
    acc_popt = []
    
    def acc_exp_func(x, a, b):
        return a*np.exp(x*b) + 5e-2

    for s in range(len(sizesvar)):

        acc_popt.append(curve_fit(acc_exp_func,  trialvar[1:]**2, accuracy[s][1:])[0])
        plt.scatter(trialvar**2, accuracy[s])
        y_fit = acc_exp_func(trialvar**2, acc_popt[s][0], acc_popt[s][1])
        plt.plot(trialvar**2, y_fit, label='N='+str(sizesvar[s]**2))

    plt.legend(fontsize=ticksize, frameon=False)
    plt.savefig('./fig1/accuracy.svg')
    plt.close() 
    
    # ----------------------- complexity curves
    for c in range(2):
        
        fig = plt.figure()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        complexity = data['se_pca_tracker']
        label = 'PCA'
        baseline = 810
        if c==1:
            complexity =data['se_tracker']
            label = 'Fourier'
            baseline = 443

        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
        plt.gca().yaxis.get_offset_text().set_fontsize(ticksize) 
        plt.xlabel('Range of lateral excitation (σ)', fontsize=fontsize)
        plt.ylabel(f'Effective dimensions ({label})', fontsize=fontsize)

        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        
        comx_popt = []

        def comx_exp_func(x, a, b, c):
            return 1e2*a*(np.exp(-x*b)+c)

        for s in range(len(sizesvar)):
            
            comx_popt.append(curve_fit(comx_exp_func,  trialvar[1:]**2, complexity[s][1:])[0])
            plt.scatter(trialvar**2, complexity[s])
            y_fit = comx_exp_func(trialvar**2, comx_popt[s][0], comx_popt[s][1], comx_popt[s][2])
            plt.plot(trialvar**2, y_fit, label='N='+str(sizesvar[s]**2))

        plt.plot(trialvar**2, [baseline]*len(trialvar), linestyle='--', color='grey')

        plt.legend(fontsize=ticksize, frameon=False)
        plt.savefig(f'./fig1/complexity_{label}.svg')
        plt.close() 
        
    # ----------------------- accuracy comxplexity tradeoff curves
    
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
    plt.xlabel('Range of lateral excitation (σ)', fontsize=fontsize)
    plt.ylabel('Accuracy / ED', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
    
    for s in range(len(sizesvar)):

        plt.scatter(trialvar**2, accuracy[s]/complexity[s], alpha=0.5)
        y_fit = acc_exp_func(trialvar**2, acc_popt[s][0], acc_popt[s][1]) / comx_exp_func(trialvar**2, comx_popt[s][0], comx_popt[s][1], comx_popt[s][2])
        plt.plot(trialvar**2, y_fit, label='pop. size='+str(sizesvar[s]**2))

    plt.legend(fontsize=ticksize, frameon=False)
    plt.savefig('./fig1/tradeoff.svg')
    plt.close() 
    
    
    # ----------------------- accuracy pinwheels
    n_pinwheels = np.array(sizesvar)[None] / ((trialvar[:,None]+lambda_popt[1]) * lambda_popt[0])
    n_pinwheels = n_pinwheels**2 * np.pi

    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Number of pinwheels', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    

    for s in range(len(sizesvar)):
        ax.scatter(n_pinwheels[:,s], accuracy.T[:,s], label='N='+str(sizesvar[s]**2))
        
    def inv_exp_func(x, a, b, c):
        return a*(1-np.exp(-(x/1e2)*b)) + c*0.1
    
    x = n_pinwheels[:].flatten()
    y = accuracy.T[:].flatten()
    
    popt, pcov = curve_fit(inv_exp_func,  x, y)
    
    pw_max = n_pinwheels.max()
    pw_min = n_pinwheels.min()
    x = np.linspace(pw_min, pw_max, 1000)
    y = inv_exp_func(x, popt[0], popt[1], popt[2])
    plt.plot(x, y, color='black', linewidth=0.5)
    
    plt.plot([pw_min, pw_max], [.30]*2, linestyle='--', color='grey')
    plt.plot([pw_min, pw_max], [.20]*2, linestyle='--', color='grey')
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    plt.legend(fontsize=ticksize, frameon=False)
    plt.savefig(f'./fig1/accuracy_pw.svg')
    plt.close()
    
    # ----------------------- tradeoff pinwheels
    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Number of pinwheels', fontsize=fontsize)
    plt.ylabel('Accuracy / ED', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    for s in range(len(sizesvar)):
        ax.scatter(n_pinwheels[1:,s], accuracy.T[1:,s]/(complexity.T[1:,s]), label='N='+str(sizesvar[s]**2))
    
    x = n_pinwheels[2:].flatten()
    y = accuracy.T[2:].flatten()
    
    ax.set_xscale('log')
        
    #popt, pcov = curve_fit(tanh_func,  x, y)
    
    #pw_max = n_pinwheels.max()/2
    #x = np.arange(pw_max)
    #plt.plot(x, tanh_func(x, popt[0], popt[1], popt[2]), color='black', linewidth=0.5)
    
    #plt.plot([0, pw_max], [.30]*2, linestyle='--', color='grey')
    #plt.plot([0, pw_max], [.20]*2, linestyle='--', color='grey')
    
    plt.legend(fontsize=ticksize, frameon=False)
    plt.savefig(f'./fig1/tradeoff_pw.svg')
    plt.close()
    
    # ----------------------- tuning curves pw
    max_pw = 45
    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Number of pinwheels', fontsize=fontsize)
    plt.ylabel('RF sharpness (a.u.)', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    
    tuning_curves = tuning_curves.max(2)[0]
    X = []
    Y = []

    for s in range(len(sizesvar)):
        x = n_pinwheels[2:,s]
        y = tuning_curves.T[2:,s]
        y = y[x<max_pw]
        x = x[x<max_pw]
        X.append(list(x))
        Y.append(list(y))
        ax.scatter(x, y, label='N='+str(sizesvar[s]**2))
        
    def linear_func(x, m, b):
        return m*x + 0.1
    
    X = [x for xs in X for x in xs]
    Y = [x for xs in Y for x in xs]
        
    #popt, pcov = curve_fit(linear_func, X, Y)
    
    x = np.array([pw_min, pw_max])
    #plt.plot(x, linear_func(x, popt[0], popt[1]), color='black', linewidth=0.5)
    
    plt.legend(fontsize=ticksize, frameon=False)
    plt.savefig(f'./fig1/tuning_pw.svg')
    plt.close()
    
    # ----------------------- complexity pinwheels
    for c in range(2):
        fig = plt.figure()
        ax = plt.gca()
        plt.subplots_adjust(bottom=0.15, left=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        complexity = data['se_tracker']
        label = 'Fourier'
        baseline = 443
        if c==1:
            complexity = data['se_pca_tracker']
            label = 'PCA'
            baseline = 810

        plt.gca().yaxis.get_offset_text().set_fontsize(ticksize) 
        plt.xlabel('Number of pinwheels', fontsize=fontsize)
        plt.ylabel(f'Effective dimensions ({label})', fontsize=fontsize)

        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        
        shifter = float(complexity.min()*0.85)

        for s in range(len(sizesvar)):
            ax.scatter(n_pinwheels[1:,s], complexity.T[1:,s]-shifter, label='N='+str(sizesvar[s]**2))

        def linear_func(x, m, b):
            return m*x + b

        x = n_pinwheels[1:].flatten()
        y = complexity.T[1:].flatten()

        popt, pcov = curve_fit(linear_func, x, y)
        
        x = np.linspace(pw_min, pw_max, 1000)
        y = linear_func(x, popt[0], popt[1]) - shifter

        plt.plot(x, y, color='black', linewidth=0.5)
        plt.plot([pw_min,pw_max], [baseline]*2, linestyle='--', color='grey')
        
        ax.set_yscale('log')
        ax.set_xscale('log')

        plt.legend(fontsize=ticksize, frameon=False)
        plt.savefig(f'./fig1/complexity_pw_{label}.svg')
        plt.close()
    
    # ----------------------- dimensions
    plt.subplots_adjust(bottom=0.15)
    for c in range(2):
        fig = plt.figure()
        ax = plt.gca()
        components = data['spectrum_tracker']
        label = 'Fourier'
        if c==1:
            components = data['comp_tracker']
            label = 'PCA'
            
        samples = 2
        size = 2
        step = len(trialvar) // samples
        
        a = 1
        if c:
            a = 4
        

        for s in range(samples):

            comp_plot = components[size, s*step, :sizesvar[size]*a, :sizesvar[size]*a]
            comp_plot /= comp_plot.max()
            comp_plot[0] = 0.5
            comp_plot[-1] = 0.5
            comp_plot[:,0] = 0.5
            comp_plot[:, -1] = 0.5
            plt.subplot(samples,1,s+1)
            plt.axis('off')
            plt.imshow(comp_plot, cmap=cm.viridis)
            
            if c == 1:
                # Calculate grid spacing
                grid_size = sizesvar[size]
                # Draw red grid lines
                for i in range(a):
                    plt.hlines(y=grid_size * i, xmin=0, xmax=sizesvar[size] * a - 1, colors='white', linewidths=0.5)
                    plt.vlines(x=grid_size * i, ymin=0, ymax=sizesvar[size] * a - 1, colors='white', linewidths=0.5)


        plt.savefig(f'./fig1/components_{(label)}.svg')            
        plt.close()
        
    # ----------------------- animals
    # Data
    labels = [
        'Owl monkey', 
        'Squirrel monkey', 
        'Crab-eating macaque', 
        'Rhesus macaque', 
        'Tree shrew', 
        'Cat',
        'Ferret',
        'Mouse Lemur'
    ]
    n_pinwheels = np.array([2124, 7007, 8720, 10152, 564, 1140, 429, 550])
    acuity = np.array([10, 41, 46, 54, 2.4, 8.85, 3.57, 3])

    # Create figure and axis
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.15)
    max_pw = 12000
    plt.xlim(0, max_pw)
    plt.ylim(0, 65)
    
    plt.xlabel('Number of pinwheels', fontsize=fontsize)
    plt.ylabel('Visual acuity (cpd)', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
    ax.yaxis.get_offset_text().set_fontsize(ticksize) 

    # Scatter plot
    scatter = ax.scatter(n_pinwheels, acuity, color='black', s=75, marker='o')
    
    def linear_func(x, m, b):
        return m*x + b
    
    popt, pcov = curve_fit(linear_func, n_pinwheels, acuity)
    x = np.array([0, max_pw])
    plot = ax.plot(x, linear_func(x, popt[0], popt[1]), linewidth=0.5, color='grey')

    # Annotating points
    texts = []
    for i, txt in enumerate(labels):
        texts.append(ax.text(n_pinwheels[i], acuity[i], txt, fontsize=20, color='dimgray'))

    # Use adjust_text to automatically adjust text positions
    adjust_text(texts)

    # Save the figure
    plt.savefig('./fig1/animals.svg')
    plt.close()
        
    sns.reset_orig()
    