import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.contrib.gp.kernels import Combination, Sum
smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
# assert pyro.__version__.startswith('1.6.0')

# https://pyro.ai/examples/gp.html

device = torch.device('cuda:0')

def train_GPR(embeddings, target, gpr_path, emb_path, y_path):
    """
    GPR training: The GPR model takes in the embeddings and the label/target of the samples
        embeddings: numpy array of shape(N, D)
        target: numpy array of shape (N, )
        
        gpr_path: the location to save the gpr model
        emb_path: the location to save the transformer oof embedding
        y_path: the location to save the original label/target
    """
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    no_steps = 1000
    ls = 3.
    var = 1.
    noise = 1.

    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(target, dtype=torch.float32).to(device)
    kernel = gp.kernels.RBF(input_dim=X.shape[1], variance=torch.tensor(var),
                        lengthscale=torch.tensor(ls)).to(device)

    gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(noise)).to(device)

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = no_steps if not smoke_test else 2
    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    with torch.no_grad():
        mean, cov = gpr(X, full_cov=True, noiseless=False)
    
    gpr_prediction = mean.cpu().numpy()
#     plt.plot(losses)
    
    print("Saving gpr model, embeddings, and target")
    save_gpr_model(gpr, embeddings, y, gpr_path, emb_path, y_path)
    
    return gpr_prediction

def save_gpr_model(gpr, embedding, y, gpr_path, embedding_path, y_path):
    """
    To save the gpr model, embeddings and the original label required to perform inference for new test points.
        gpr: pyro gpr model
        embedding: transformer oof embedding
        y: original label/target
        
        gpr_path: the filepath to save the gpr model
        embedding_path: the filepath to save the transformer oof embedding
        y_path: the filepath to save the original label/target
        
    """    
    torch.save(gpr.state_dict(), gpr_path)
    np.save(embedding_path, embedding)
    np.save(y_path, y.cpu())
    print("Finished saving gpr weights at: {}".format(gpr_path))
    print("Finished saving embeddings at: {}".format(embedding_path))
    print("Finished saving embeddings label/target at: {}".format(y_path))
    

def get_pyro_emb_preds(embedding_path, y_path, gpr_model_path, test_embedding):
    """
    GPR inference: Reads in the embedding file, label/target file, gpr model weights
        embedding_path: The saved embeddings in .npy format
        y_path: The saved label/target in .npy format
        gpr_model_path: The saved pyro GPR model        
        test_embedding: The transformer embedding (N, D)
    """
    pyro.clear_param_store()
    emb_train = np.load(embedding_path, allow_pickle=True)
    X = torch.tensor((emb_train), dtype=torch.float32).to(device)
    y =  torch.tensor(np.load(y_path), dtype=torch.float32).to(device)
 
    emb_test = np.vstack(test_embedding)
    X_test = torch.tensor((emb_test), dtype=torch.float32).to(device)
    kernel = gp.kernels.RBF(input_dim=X.shape[1], variance=torch.tensor(1.),
                    lengthscale=torch.tensor(1.)).to(device)
    gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(1.)).to(device)
    gpr.load_state_dict(torch.load(gpr_model_path))
    
    with torch.no_grad():
        mean, cov = gpr(X_test, full_cov=True, noiseless=False)
    mean = np.array(mean.cpu())
    return mean


'''
### Example usage
############# Training GPR model #######################

## The following variables are the filepath for saving the GPR model, the label/target, and the transformer embeddings
gpr_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/gpr_model'
y_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/target.npy'
emb_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/embedding.npy'

## train a GPR on the transformer embeddings
mean_pred = train_GPR(embeddings, y, gpr_path, emb_path, y_path)
'''