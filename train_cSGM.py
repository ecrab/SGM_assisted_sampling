import numpy as np
import torch
import os

import utils
import models

def train(xs_in, labels_in, epochs, batch_size, kappa_bound, model_dir_in):
    N_epochs = epochs
    train_size = xs_in.shape[0]
    batch_size = 20
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    
    xs = xs_in
    labels = labels_in
    score_model = models.ScoreNet()
    kappa = kappa_bound
    R = 1000
    
    model_dir = model_dir_in
    
    #Initialize the optimizer
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)
    
    for k in range(N_epochs):
        losses = []

        batch_target_labels = np.random.choice(labels, size=batch_size, replace=True)
        batch_real_indx = np.zeros(batch_size, dtype=int)
        for j in range(batch_size):
            indx_real_in_vicinity = np.where(np.abs(labels-batch_target_labels[j])<= kappa)[0]
            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1, replace=True)[0]

        batch = xs[batch_real_indx]
        yt = labels[batch_real_indx]
        yt = torch.from_numpy(yt).type(torch.float)
        N_batch = batch.shape[0]
        t1 = np.random.randint(1, R, (N_batch,1))/(R-1)
        t = torch.from_numpy(t1)
        mean_coeff = utils.mean_factor(t)
        vs = utils.var(t)
        stds = torch.sqrt(vs)
        noise = torch.from_numpy(np.random.normal(size=batch.shape))
        xt = batch * mean_coeff + noise * stds
        optimizer.zero_grad()
        output = score_model(xt.float(), t.float(), yt.float())
        loss_fn = torch.mean((noise + output*vs)**2)
        loss_fn.backward()
        optimizer.step()
        loss_fn_np = loss_fn.detach().cpu().numpy()
        losses.append(loss_fn_np)
        mean_loss = np.mean(np.array(losses))

        
        
        x_bin = np.linspace(-1.5, 1.5, 51)
        y = torch.full((500,), 5)
        trained_score = lambda x, t: score_model(x.float(), t.float(), y.float())
        step_rng = np.random.default_rng()
        samples = utils.reverse_sde(step_rng, 2, 500, utils.drift, utils.diffusivity, trained_score)
        est_pdf, est_bins = np.histogram(np.hstack(samples[:,1]), bins=x_bin, density=True)
        dbin = est_bins[1] - est_bins[0]
        x = (est_bins + dbin / 2)[:-1]
        l1_norm = np.trapz(np.abs(est_pdf - true_pdf(x)), x=x)


        if k % 1000 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            model_save = os.path.join(model_dir, f'global_step_{k:06d}.pth')
            torch.save(score_model, model_save)

        if l1_norm < 0.4:
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            model_save = os.path.join(model_dir, f'global_step_{k:06d}.pth')
            torch.save(score_model, model_save)
            print("Epoch %d \t, loss %f, norm %f " % (k, mean_loss, l1_norm))
        
    return score_model
        


def potential(x):
    k=5
    return (x**2 - 1)**2 + (0.2*k - 1)*x

def true_pdf(x):
    exp_minus_potential_x = np.exp(-potential(x))
    return exp_minus_potential_x / np.trapz(exp_minus_potential_x, x=x)