from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from typing import Literal, Tuple, Optional, Any
import random
import json
import os


class ObjaverseDataset(Dataset):
    def __init__(self,
        root_dir: str,
        img_wh: Tuple[int, int],
        object_json: str,
        validation: bool = False,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        # trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        read_normal: bool = False,
        read_depth: bool = False,
        suffix: str = 'jpg',
        subscene_tag: int = 2,
        backup_scene: str = "3f64480e3b1c48139919a3da331f8c5f"
        ) -> None:

        self.root_dir = Path(root_dir)
        self.validation = validation
        self.num_samples = num_samples
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_depth = read_depth
        self.suffix = suffix
        self.subscene_tag = subscene_tag

        self.view_types = ["flat_view", "flat_view_var", "top_view", "bottom_view"]
        self.view_prompt = {
            "flat_view": "4 orthogonal flat views of ", 
            "flat_view_var": "4 orthogonal flat side views of ", 
            "top_view": "4 orthogonal top side views of ", 
            "bottom_view": "4 orthogonal bottom side views of "
        }
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_wh, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution),
            # transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if object_json is not None:
            with open(object_json) as f:
                glbs = json.load(f)
            self.objects = sorted(glbs)
            self.prompt = [glbs[k] for k in self.objects]
            
        else:
            raise ValueError("object_json is None")

        if not validation:
            self.objects = self.objects[:-num_validation_samples]
            self.prompt = self.prompt[:-num_validation_samples]
        else:
            self.objects = self.objects[-num_validation_samples:]
            self.prompt = self.prompt[-num_validation_samples:]
        if num_samples is not None:
            self.objects = self.objects[:num_samples]
            self.prompt = self.prompt[:num_samples]

        print("loading ", len(self.objects), " objects and prompts in the dataset")

        self.backup_data = self.__loaditem__(0, backup_scene, glbs[backup_scene]) 

    def __len__(self):
        return len(self.objects)

    def load_mask(self, img_path,return_type='np'):
        depth_img = np.array(Image.open(img_path).resize(self.img_wh))
        depth_min = np.min(depth_img)
        valid_mask = depth_img != depth_min
        invalid_mask = depth_img == depth_min
        invalid_mask = invalid_mask.astype(np.uint8) * 255

        if return_type == "np":
            pass
        elif return_type == "pt":
            mask = torch.from_numpy(invalid_mask)
        else:
            raise NotImplementedError
        
        return mask
    
    def load_image(self, img_path):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = Image.open(img_path)
        transformed_img = self.train_transforms(img) # [0, 1]
        # img = np.array(Image.open(img_path).resize(self.img_wh))
        # img = img.astype(np.float32) / 255. 
        assert transformed_img.shape[0] == 3 or transformed_img.shape[0] == 4 # RGB or RGBA
        
        return transformed_img
    
    def __loaditem__(self, index, debug_object=None, debug_prompt=None):
        if debug_object is not None:
            object_name =  debug_object #
            prompt = debug_prompt
        else:
            object_name = self.objects[index%len(self.objects)]
            prompt = self.prompt[index%len(self.prompt)]
        
        view_types = self.view_types

        render_dir = os.path.join(self.root_dir,  object_name[:self.subscene_tag].lower(), object_name)

        
        types = len(os.listdir(render_dir)) // 3
        # select a random view
        view_type = random.choice(view_types[:types])
        view_prompt = self.view_prompt[view_type]
        
        img_path = os.path.join(render_dir, "rgb_%s.%s" % (view_type, self.suffix))
        normal_path = os.path.join(render_dir, "normals_%s.%s" % (view_type, self.suffix))
        depth_path = os.path.join(render_dir, "depth_%s.%s" % (view_type, self.suffix))

        img_out = self.load_image(img_path)

        if self.read_depth:
            cond_in = self.load_image(depth_path)
        elif self.read_normal:
            cond_in = self.load_image(normal_path)
        else:
            # cond_in = None
            return {
            'prompt': view_prompt + prompt,
            'img_out': img_out,
        }

        return {
            'prompt': view_prompt + prompt,
            'cond_in': cond_in,
            'img_out': img_out,
        }

    def __getitem__(self, index):
        try:
            data = self.__loaditem__(index)
            if data is None:
                print("load error ", self.objects[index%len(self.objects)] )
                return self.backup_data
            return data
        except:
            print("load error ", self.objects[index%len(self.objects)] )
            return self.backup_data
        

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.num_datasets = len(datasets)

    def __getitem__(self, i):
        chosen = random.choices(self.datasets, self.weights, k=1)[0]
        return chosen[i]

    def __len__(self):
        return max(len(d) for d in self.datasets)

if __name__ == "__main__":
    train_dataset = ObjaverseDataset(
        root_dir="WonderTex/data_lists/rendering",
        object_json="WonderTex/data_lists/data_prepare/finished_automated_Objaverse_no3Dword.json",
        img_wh=(512, 512),
        num_validation_samples=1,
        validation=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=5
    )
    for step, batch in enumerate(train_dataloader):
        print(step)
        print(batch)
        break
    print(len(train_dataloader))
    data1  = train_dataset[3]
    # print(data0["img_out"].shape)
    # print(data0["img_out"].max())
    # print(data0["img_out"].min())
    print(data1["prompt"])
    print(len(train_dataset))

    """
    loading  25080  objects and prompts in the dataset
    torch.Size([3, 512, 512])
    tensor(0.6706)
    tensor(-1.)
    4 orthogonal flat views of a labeled clam shell with a ruler next to it.
    """
