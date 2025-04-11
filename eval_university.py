import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model.get_model import get_model


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

@dataclass
class Configuration:
    # Model
    model: str = 'DINOv2' # 'DINOv2' # ConvNext
    arch_name: str = 'dinov2_vitb14'
    num_trainable_blocks: int = 4

    # Aggregator
    aggregator_name: str = 'salad' #'salad' #'gem'
    num_channels: int = DINOV2_ARCHS[arch_name]
    num_clusters: int = 64
    cluster_dim: int = 128
    token_dim: int = 256
    
    # Override model image size
    img_size: int = 518 # 384 for ConvNext 518 for DINOv2
    
    # Evaluation
    batch_size: int = 10
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = False if model == 'SaliencyCVGL' else True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-S2S'           # 'U1652-D2S' | 'U1652-S2D' | 'U1652-S2S'
    
    # Checkpoint to start from
    checkpoint_start: str = '/workspace/WorkSpacePR/Sample4Geo/run_logs/DINOv2_U1652-S2S/153041/weights_end.pth'

    # Outpath
    model_path: str = "/workspace/WorkSpacePR/Sample4Geo/run_logs/DINOv2_U1652-S2S/153041"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
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
    config.query_folder_acmm_test = '/public/University-1652/masked_test_set/University-1652_mask/workshop_query_street'
    config.gallery_folder_acmm_test = '/public/University-1652/masked_test_set/University-1652_mask/workshop_gallery_satellite'
else:
    raise NotImplementedError(f'Sorry, <{config.dataset}> dataset is not implemented!')


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))
    model, mean, std = get_model(config)

    img_size = (config.img_size, config.img_size)

    # load pretrained Checkpoint    
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

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
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
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    print("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       outpath=config.model_path,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)