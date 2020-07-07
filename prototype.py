from tqdm import tqdm
import fire

import copy
import time

import numpy as np
import scipy as sp
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from model import GAT_MNIST
import util

to_cuda = util.to_cuda

def train_model(
        epochs,
        batch_size,
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    print("Reading dataset")
    dset = CIFAR10(dset_folder, download=True, train=True, transform=util.get_graph_from_image)

    valid_split = 0.1
    valid_len = int(len(dset) * valid_split)
    train_len = len(dset) - valid_len
    dset_train, dset_valid = torch.utils.data.random_split(dset, [train_len,valid_len])

    dset_train_loader, dset_valid_loader = map(
        lambda ds: torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=util.batch_graphs,),
        [dset_train,dset_valid]
    )
    
    model_args = []
    model_kwargs = {}
    model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    if use_cuda:
        model = model.cuda()
    
    opt = torch.optim.Adam(model.parameters())
    
    best_valid_acc = 0.
    best_model = copy.deepcopy(model)
    
    last_epoch_train_loss = 0.
    last_epoch_train_acc = 0.
    last_epoch_valid_acc = 0.
    
    interrupted = False
    for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
        try:
            train_losses, train_accs = util.train(model, opt, dset_train_loader, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
            
            last_epoch_train_loss = np.mean(train_losses)
            last_epoch_train_acc = 100*np.mean(train_accs)
        except KeyboardInterrupt:
            print("Training interrupted!")
            interrupted = True
        
        valid_accs = util.test(model, dset_valid_loader, use_cuda, desc="Validation ", disable_tqdm=disable_tqdm,)
                
        last_epoch_valid_acc = 100*np.mean(valid_accs)
        
        if last_epoch_valid_acc>best_valid_acc:
            best_valid_acc = last_epoch_valid_acc
            best_model = copy.deepcopy(model)
        
        tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
        
        if interrupted:
            break
    
    util.save_model("best",best_model)
    util.save_model("last",model)


def test_model(
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    best_model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    util.load_model("best",best_model)
    if use_cuda:
        best_model = best_model.cuda()
    
    test_dset = CIFAR10(dset_folder, download=True, train=False, transform=util.get_graph_from_image)
    dset_test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=True, num_workers=2, collate_fn=util.batch_graphs,)
    
    test_accs = util.test(best_model, dset_test_loader, list(range(len(test_labels))), use_cuda, desc="Test ", disable_tqdm=disable_tqdm,)
    test_acc = 100*np.mean(test_accs)
    print("TEST RESULTS: {acc:.2f}%".format(acc=test_acc))

    

def main(
        train:bool=False,
        test:bool=False,
        epochs:int=100,
        batch_size:int=32,
        use_cuda:bool=True,
        disable_tqdm:bool=False,
        dset_folder:str = "./cifar10"
        ):
    use_cuda = use_cuda and torch.cuda.is_available()
    if train:
        train_model(
                epochs = epochs,
                batch_size = batch_size,
                use_cuda = use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )
    if test:
        test_model(
                use_cuda=use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )

if __name__ == "__main__":
    fire.Fire(main)
