import numpy as np
import torch
import os

import utils
import models

def train(xs_in, labels_in, epochs, batch_size, kappa_bound, model_dir_in):
    N_epochs = epochs
    train_size = xs_in.shape[0]
    batch_size = 200
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
        #batch = batch.reshape((-1,1))
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

        if k % 100 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            model_save = os.path.join(model_dir, f'global_step_{k:06d}.pth')
            torch.save(score_model, model_save)

        
    return score_model
        

