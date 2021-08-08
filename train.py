import os
import time
import tqdm
import threading

import torch
from torch.utils.tensorboard import SummaryWriter

from opt.base import BaseOpt

from models.classifier import Classifier
from dataset.mnist import MNist


if __name__ == "__main__":
    
    # === Init opt === #
    opt = BaseOpt('./opt/classifier.ini').get_args()
    
    # === Init Device === #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device : {device} !!!")
    
    # === Define model & Load last weights === #
    model = Classifier(opt).to(device)
    initial_epoch = model.load_weights()
    
    # === Define DataLoader === #
    train_dl = MNist(opt, opt.train).data_loader()
    valid_dl = MNist(opt, opt.valid).data_loader()
    
    # === Tensorboard SummaryWriter === #
    writer = SummaryWriter(log_dir=opt.log_dir)
    def run_tensorboard(log_dir):
        os.system(f"tensorboard --logdir={log_dir}")
    th = threading.Thread(target=run_tensorboard, args=(opt.log_dir, ), daemon=True)
    th.start()
    time.sleep(0.1)
    
    # === Training Loop === #
    print(f"Start training from epoch {initial_epoch}!")
    for epoch in range(initial_epoch, opt.epochs):
        
        # === Training === #
        model.train()
        train_losses = []
        pgbar = tqdm.tqdm(train_dl)
        for x, target in pgbar:
            loss = model.train_step(x.to(device), target.to(device))
            train_losses.append(loss)
            
            pgbar.set_description(f"epoch {epoch}/{opt.epochs} : loss : {loss}")
        
        train_loss = sum(train_losses) / len(train_losses)
        writer.add_scalar("Loss/train", train_loss, global_step=epoch)
        
        # === Validation === #
        model.eval()
        val_losses = []
        for x, target in pgbar:
            with torch.no_grad():
                loss = model.forward_with_loss(x.to(device), target.to(device))
            val_losses.append(loss)
        
        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar("Loss/val", val_loss, global_step=epoch)
        print(f"validation loss : {val_loss}\n")
        
        # === Save Weights === #
        model.save_weights()
        
        writer.flush()
        time.sleep(0.1)
    
    writer.close()
        