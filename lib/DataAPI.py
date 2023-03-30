import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
 
class deepglobledataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载所有图像路径，并按照名称排序已确定掩膜对应正确

        self.imgs = self.loadData('sat')
        self.masks = self.loadData('mask')

    def loadData(self,type):
        datalist = list(sorted(os.listdir(self.root)))
        reslist = []
        if type == 'mask':
            for i in datalist:
                if i.endswith('mask.png'):
                    reslist.append(i)
        else:
            for i in datalist:
                if i.endswith('sat.jpg'):
                    reslist.append(i)
        return reslist

    def __getitem__(self, idx):
        # 相图像加载掩膜
        img_path = os.path.join(self.root, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 掩膜无需转化为RGB模式
        # 每种颜色对应不同实例，0为背景
        mask = Image.open(mask_path).convert('1')

        mask = np.array(mask)
        # 实例编码为不同颜色
        obj_ids = np.unique(mask)
        # 第一个编码为背景，去除
        obj_ids = obj_ids[1:]
 
        # 分离掩膜编码为二进制编码
        masks = mask == obj_ids[:, None, None]
 
        # 得到每个掩膜的BB
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            a = masks[i]
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 单类时
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例不重叠
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)
