import numpy as np
import os
from config import get_dir

def gen(dataset_name, save_class_names_path):
    if dataset_name == 'cub':
        
        from config import cub_root
        import re
        
        classes_file = os.path.join(cub_root, "CUB_200_2011/classes.txt")
        with open(classes_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        classnames = [re.split(r'\d+\s+\d+\.', line.strip())[1] for line in lines]
        assert len(classnames) == 200
    
    elif dataset_name == 'scars': 
        from config import car_root as car_dir
        import scipy.io as sio
        car_root = car_dir + "/cars_{}/"
        meta_default_path = car_dir + "/cars_{}.mat"
        
        meta_file = car_dir + "/devkit/cars_meta.mat"
        meta_data = sio.loadmat(meta_file)
        classnames = [name[0] for name in meta_data['class_names'][0]]
        assert len(classnames) == 196
        
    elif dataset_name == 'aircraft': 
        from config import aircraft_root
        
        ## only variant
        class_type = 'variant'  
        split = 'train'  

        classes_file = os.path.join(aircraft_root, 'data', f'images_{class_type}_{split}.txt')

        classes = set()
        with open(classes_file, 'r') as f:
            for line in f:
                class_name = ' '.join(line.split(' ')[1:]).strip()
                classes.add(class_name)
        classnames = sorted(classes)  
        assert len(classnames) == 100
        
    elif dataset_name == 'cifar10': 
        from torchvision.datasets import CIFAR10
        from config import cifar_10_root
        cifar10_dataset = CIFAR10(root=cifar_10_root, train=True, download=False)
        classnames = cifar10_dataset.classes
        assert len(classnames) == 10
        
    elif dataset_name == 'cifar100': 
        from torchvision.datasets import CIFAR100
        from config import cifar_100_root
        cifar100_dataset = CIFAR100(root=cifar_100_root, train=True, download=False)
        classnames = cifar100_dataset.classes
        assert len(classnames) == 100
        
    elif dataset_name == 'imagenet_1k': 
        
        ## download imagenet classnames
        ## !wget -O imagenet_classes.txt "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

        with open(os.path.join(get_dir, "dataset_class_name/imagenet_classes.txt"), "r") as f:
            classnames = [line.strip() for line in f.readlines()]

        assert len(classnames) == 1000
    
    elif dataset_name == 'imagenet_100':
        np.random.seed(0)  
        imagenet_100_classes_idx = np.random.choice(range(1000), size=(100,), replace=False)
        imagenet_100_classes_idx = np.sort(imagenet_100_classes_idx)  
        with open(os.path.join(get_dir, "dataset_class_name/imagenet_classes.txt"), "r") as f:
            imagenet_1k_classnames = [line.strip() for line in f.readlines()]
        classnames = [imagenet_1k_classnames[i] for i in imagenet_100_classes_idx]
        
        assert len(classnames) == 100
        

    
    elif dataset_name == 'herbarium_19':

        import pandas as pd

        herb_classes_path = os.path.join(get_dir, "dataset_class_name/herb_categories.txt")
        df = pd.read_csv(herb_classes_path)
        classnames = df["ScientificName"].tolist()
        assert len(classnames) == 683
    
    else:

        raise NotImplementedError

    
        

    
    np.save(save_class_names_path,np.array(classnames))
    
    return

if __name__ == "__main__":
    gen("cub", None)