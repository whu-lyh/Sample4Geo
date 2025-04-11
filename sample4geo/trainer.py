import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from .utils import AverageMeter


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()
    losses = AverageMeter()
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    step = 1
    accum_steps = int(train_config.accum_steps)
    accum_steps = (accum_steps if accum_steps > 1 else 1)
    
    bar = tqdm(dataloader, total=len(dataloader)) if train_config.verbose else dataloader
    
    # for loop over one epoch
    for iteration, (query, reference, ids) in enumerate(bar):
        if scaler:
            with autocast():
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                # Forward pass
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        else:
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss = loss / accum_steps
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(dataloader):
                # Update model parameters (weights)
                optimizer.step()
                # Zero gradients for next step
                optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item() * accum_steps),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1
    return losses.avg

def predict(train_config, model, dataloader, is_query=False):
    model.eval()
    
    bar = tqdm(dataloader, total=len(dataloader)) if train_config.verbose else dataloader

    device = train_config.device
    normalize_features = train_config.normalize_features

    # Get dataset size to preallocate memory if possible
    dataset_size = len(dataloader.dataset)
    feature_dim = model.feature_dim  # Ensure your model has this attribute

    # Preallocate memory for results (avoid list appends)
    img_features = torch.zeros((dataset_size, feature_dim), dtype=torch.float32, device=device)
    ids_list = torch.zeros(dataset_size, dtype=torch.long, device=device)  # Assuming ids are integers

    with torch.no_grad():
        index = 0
        for img, ids in bar:
            batch_size = img.size(0)
            img = img.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                if train_config.dual_mode:
                    if is_query:
                        img_feature = model.fetch_feat_stre(img)
                    else:
                        img_feature = model.fetch_feat_sate(img)
                else:
                    img_feature = model(img)

            if normalize_features:
                img_feature = F.normalize(img_feature.float(), dim=-1)

            # Store directly into preallocated tensors
            img_features[index : index + batch_size] = img_feature.detach().to(torch.float32)
            ids_list[index : index + batch_size] = ids.detach()

            index += batch_size  # Update index
    
    return img_features, ids_list

def vanilla_predict(train_config, model, dataloader):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    ids_list = []
    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation, list append operation will cause huge memory occupation
            img_features_list.append(img_feature.detach().to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list