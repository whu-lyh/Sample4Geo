import math
import os
import shutil
import sys
import time
from dataclasses import dataclass

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)

from sample4geo.dataset.university import (U1652DatasetEval, U1652DatasetTrain,
                                           get_transforms)
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model.get_model import get_model
from sample4geo.trainer import train
from sample4geo.utils import Logger, setup_system

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

@dataclass
class Configuration:
    # Model
    model: str = 'ConvNext' # 'SaliencyCVGL' # 'DINOv2' # ConvNext
    dual_mode: bool = False # for SaliencyCVGL, False for shared backbone
    arch_name: str = 'dinov2_vitb14'
    num_trainable_blocks: int = 2

    # Aggregator
    aggregator_name: str = 'netvlad' #'salad' #'gem'
    num_channels: int = DINOV2_ARCHS[arch_name]
    num_clusters: int = 64
    cluster_dim: int = 128
    token_dim: int = 256
    
    # Override model image size
    img_size: int = 384 # 384 for ConvNext or 518 for DINOv2
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 3
    batch_size: int = 10                # keep in mind real_batch_size = 2 * batch_size, 10 for ConvNext 16 for DINOv2, 64 for SaliencyCVGL
    verbose: bool = True
    gpu_ids: tuple = (0)#,1,2,3)        # GPU ids for training
    accum_steps: int = 128              # accumulation steps    
    
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.2
    
    # Learning Rate
    lr: float = 0.0005                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"             # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0000001             #  only for "polynomial"
    
    # Dataset
    dataset: str = 'U1652-S2D'           # 'U1652-D2S' | 'U1652-S2D' | 'U1652-S2S'

    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "./run_logs"

    # Save predictions
    predict_file_path: str = ""

    # Eval before training
    zero_shot: bool = True

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 8 

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/public/University-1652/train/satellite'
    config.gallery_folder_train = '/public/University-1652/train/drone'   
    config.query_folder_test = '/public/University-1652/test/query_drone' 
    config.gallery_folder_test = '/public/University-1652/test/gallery_satellite'    
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = '/public/University-1652/train/satellite'
    config.gallery_folder_train = '/public/University-1652/train/drone'    
    config.query_folder_test = '/public/University-1652/test/query_satellite'
    config.gallery_folder_test = '/public/University-1652/test/gallery_drone'
elif config.dataset == 'U1652-S2S':
    config.query_folder_train = '/public/University-1652/train/street'
    config.gallery_folder_train = '/public/University-1652/train/satellite'    
    config.query_folder_test = '/public/University-1652/test/query_street'
    config.gallery_folder_test = '/public/University-1652/test/gallery_satellite'
    # actually the same as test set but reorganized for better submission test
    config.query_folder_acmm_test = '/public/University-1652/masked_test_set/University-1652_mask/workshop_query_street'
    config.gallery_folder_acmm_test = '/public/University-1652/masked_test_set/University-1652_mask/workshop_gallery_satellite'
else:
    raise NotImplementedError(f'Sorry, <{config.dataset}> dataset is not implemented!')


if __name__ == '__main__':
    # Ensure the output dir
    model_path = "{}/{}_{}/{}".format(config.model_path, config.model, config.dataset, time.strftime("%H%M%S"))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))
    model, mean, std = get_model(config)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)
    img_size = (config.img_size, config.img_size)
    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    # Train
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps) 
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))
        r1_test = evaluate(config=config,
                           outpath=model_path,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0

    for epoch in range(1, config.epochs+1):
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]['lr']))
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
            r1_test = evaluate(config=config, outpath=model_path,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            if r1_test > best_score:
                best_score = r1_test
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_r@1_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_r@1_{:.4f}.pth'.format(model_path, epoch, r1_test))
                
        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
