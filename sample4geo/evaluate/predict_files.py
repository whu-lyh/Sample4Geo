
import os

import numpy as np
import torch

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset_160k_sat(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, os.path.splitext(fname)[0])
                images.append(item)
    return images

def get_SatId_160k(img_path):
    labels = []
    paths = []
    for path, v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

def get_result_rank10(qf, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10

def save_predicts_for_submission(query_feature, gallery_feature, gallery_path_base: str, save_filename: str):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
    gallery_path = make_dataset_160k_sat(gallery_path_base, IMG_EXTENSIONS)
    gallery_label, gallery_path  = get_SatId_160k(gallery_path)
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    results_rank10 = []
    print("The number of query_feature in acmm test set: ", len(query_feature))
    gallery_label = np.array(gallery_label)
    for i in range(len(query_feature)):
        result_rank10 = get_result_rank10(query_feature[i], gallery_feature, gallery_label)
        results_rank10.append(result_rank10)

    results_rank10 = np.row_stack(results_rank10)
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    with open(save_filename, 'w') as f:
        for row in results_rank10:
            f.write('\t'.join(map(str, row)) + '\n')