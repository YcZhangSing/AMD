import concurrent.futures
import io

import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from random import random as rand
import re
###DGM4
from distutils.command.config import config
import json
from torch.utils.data import Dataset
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import os
from torchvision.transforms.functional import hflip, resize
import math

import numpy as np
import sys
sys.path.append('/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/florence2-finetuning')
from DatasetUtils import pre_caption
describles_answ = {}

describe_temple = "The following are multiple choice questions about fake news detection. \n\nThe caption of news is: "
describe_ques_latter = ". The identity and emotion of the face, and the semantic and sentiment of the text should not be manipulated. Question: Is there any face swap/attribute or text_swap in the news?\nA. No.\nB. Only face swap.\nC. Only face attribute.\nD. Only text swap.\nE. Face swap and text swap.\nF. Face attribute and text swap.\nThe options is:"

describles_answ['orig'] = "A. No."
describles_answ['face_swap'] = "B. Only face swap." # [0, 1, 0, 0, 0, 0]
describles_answ['face_attribute'] = "C. Only face attribute." #[0, 0, 1, 0, 0, 0]
describles_answ['text_swap'] = "D. Only text swap."#[0, 0, 0, 1, 0, 0]
describles_answ['face_swap&text_swap'] = "E. Both face swap and text swap." #[0, 0, 0, 0, 1, 0]
describles_answ['face_attribute&text_swap'] = "F. Both face attribute and text swap."#[0, 0, 0, 0, 0, 1]

face_locate = "If there is manipulation of a face, locate the most likely manipulated face in the image and append the results to your selected option.\nThe answer is:"
face_text_locate = "If there is manipulation of a face, locate the most likely manipulated face in the image and append the results to your selected option. If there is text_swap, list all swapped words in the caption.\nThe answer is:"

describe_ques_latter_OB = ". The identity and emotion of the face, and the semantic and sentiment of the text should not be manipulated. Question: Is there any fake face or fake words in the news?\nA. No.\nB. Yes.\nThe options is:"




class DGM4_Dataset(Dataset):
    '''在我们的类DGM4数据集上的dataset类
    '''

    def __init__(self, split, data, max_words=30, image_res=224):
        self.name = "DGM4"
        
        self.data = data
        # self.transform = transform
        self.max_words = max_words
        self.image_res = image_res

        is_train = False
        if split == 'train':
            is_train = True
        
        self.is_train = is_train # 如果是is_train，那么会在训练集中随机加入flip增强
        self.task_prompt = "<VQA>"
        

    def __len__(self):
        return len(self.data)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)
    
    def denormalize_fake_image_box_xyxy(self, fake_image_box, image_width, image_height):
        """
        Converts a normalized fake_image_box in [center_x, center_y, w, h] format 
        to absolute xyxy coordinates with two decimal precision.
        
        Args:
            fake_image_box (torch.Tensor): Tensor containing normalized [center_x, center_y, w, h].
            image_width (int): The width of the original image.
            image_height (int): The height of the original image.
        
        Returns:
            tuple: (x1, y1, x2, y2) in absolute coordinates, rounded to two decimal places.
        """
        # Unpack the normalized coordinates
        center_x, center_y, w, h = fake_image_box

        # Convert normalized coordinates to absolute values
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_w = w * image_width
        abs_h = h * image_height

        # Calculate xyxy format
        x1 = abs_center_x - abs_w / 2
        y1 = abs_center_y - abs_h / 2
        x2 = abs_center_x + abs_w / 2
        y2 = abs_center_y + abs_h / 2

        # Round to two decimal places and return as a tuple
        return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)

    def __getitem__(self, index):

        ann = self.data[index]
        label = ann['fake_cls']
        
        img_dir = ann['image']

        image_dir_all = img_dir
        
        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        mask = np.zeros((self.image_res,self.image_res,1))
        
        if any(keyword in label for keyword in ['face_swap', 'face_attribute']):
            try:
                x, y, w, h = self.get_bbox(ann['fake_image_box'])
                has_bbox = True
            except Exception:
                fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        else:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
            
                

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        # image = self.transform(image)

        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)
            
            mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, :] = 1

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                                           center_y / self.image_res,
                                           w / self.image_res,
                                           h / self.image_res],
                                          dtype=torch.float)


        caption = pre_caption(ann['text'], self.max_words)
        # 原始代码
        # fake_text_pos = ann['fake_text_pos']
        # 修改后 如果获取不到fake_text_pos，则fake_text_pos的值为[]，
        fake_text_pos = ann.get('fake_text_pos', [])


        fake_text_pos_list = torch.zeros(self.max_words)
        mask = torch.tensor(mask[None, ..., 0]).float()

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1


        # conversation = []
        # conversation.append(
        #     {"from": "human", "value": describe_temple + caption + describe_ques_latter})
        # conversation.append({"from": "gpt", "value": describles_answ[label]})
        
        question = '<DGM4>'+describe_temple + caption + describe_ques_latter + face_locate
        answer = describles_answ[label]
        
        if has_bbox:
            ## florence2返回的坐标是xyxy格式的 x1,y1,x2,y2 = 365.4,465.2,765.8,999.6
            x1,y1,x2,y2 = self.denormalize_fake_image_box_xyxy(fake_image_box,W,H)
            # 保留两位小数，并插入到字符串模板中
            face_bbox_answer = (
                "Manipulated face"
                + f"<loc_{int(x1)}>"
                + f"<loc_{int(y1)}>"
                + f"<loc_{int(x2)}>"
                + f"<loc_{int(y2)}>"
            )
            answer += face_bbox_answer
        # conversation = '<DGM4>'+conversation
        # mask是根据边界框生成一个与目标分辨率 (self.image_res) 相同大小的掩码（mask），对应的区域被赋值为 1，其余为 0。
        # return image, question, answer,label, caption, fake_image_box, fake_text_pos_list, W, H, mask
        return image, question, answer,fake_image_box

###########################


class OriDGM4Dataset(Dataset):
    '''用于原始的DGM4数据集的训练
        即包含的answer中有Fake words
    '''

    def __init__(self, split, data, max_words=30, image_res=224):
        self.name = "DGM4"
        
        self.data = data
        # self.transform = transform
        self.max_words = max_words
        self.image_res = image_res

        is_train = False
        if split == 'train':
            is_train = True
        
        self.is_train = is_train # 如果是is_train，那么会在训练集中随机加入flip增强
        self.task_prompt = "<VQA>"
        

    def __len__(self):
        return len(self.data)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)
    
    def denormalize_fake_image_box_xyxy(self, fake_image_box, image_width, image_height):
        """
        Converts a normalized fake_image_box in [center_x, center_y, w, h] format 
        to absolute xyxy coordinates with two decimal precision.
        
        Args:
            fake_image_box (torch.Tensor): Tensor containing normalized [center_x, center_y, w, h].
            image_width (int): The width of the original image.
            image_height (int): The height of the original image.
        
        Returns:
            tuple: (x1, y1, x2, y2) in absolute coordinates, rounded to two decimal places.
        """
        # Unpack the normalized coordinates
        center_x, center_y, w, h = fake_image_box

        # Convert normalized coordinates to absolute values
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_w = w * image_width
        abs_h = h * image_height

        # Calculate xyxy format
        x1 = abs_center_x - abs_w / 2
        y1 = abs_center_y - abs_h / 2
        x2 = abs_center_x + abs_w / 2
        y2 = abs_center_y + abs_h / 2

        # Round to two decimal places and return as a tuple
        return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)

    
    def extract_swapped_words(self, caption, fake_text_pos):
        """
        根据 fake_text_pos 提取 caption 中的 swapped words，并返回格式化字符串。

        参数：
        caption (str): 新闻的标题文本。
        fake_text_pos (list): 一个包含被篡改单词索引的列表（从0开始）。

        返回：
        str: 以 "Swapped words: xxx, xxx, xxx" 格式返回被篡改单词字符串。
        """
        words = caption.split()  # 按空格拆分 caption 成单词列表
        swapped_words = [words[i] for i in fake_text_pos if 0 <= i < len(words)]  # 提取被篡改的单词
        return f" Swapped words: {', '.join(swapped_words)}" if swapped_words else "Swapped words: None"


    
    def __getitem__(self, index):

        ann = self.data[index]
        label = ann['fake_cls'].replace('text_attribute','text_swap')

        
        img_dir = os.path.join('/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/',ann['image'])

        image_dir_all = img_dir
        
        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        mask = np.zeros((self.image_res,self.image_res,1))
        
        if any(keyword in label for keyword in ['face_swap', 'face_attribute']):
            try:
                x, y, w, h = self.get_bbox(ann['fake_image_box'])
                has_bbox = True
            except Exception:
                fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        else:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
                

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        # image = self.transform(image)

        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)
            
            mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, :] = 1

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                                           center_y / self.image_res,
                                           w / self.image_res,
                                           h / self.image_res],
                                          dtype=torch.float)


        caption = pre_caption(ann['text'], self.max_words)
        # 原始代码
        # fake_text_pos = ann['fake_text_pos']
        # 修改后 如果获取不到fake_text_pos，则fake_text_pos的值为[]，
        fake_text_pos = ann.get('fake_text_pos', [])


        fake_text_pos_list = torch.zeros(self.max_words)
        mask = torch.tensor(mask[None, ..., 0]).float()

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1



        
        question = '<DGM4>'+describe_temple + caption + describe_ques_latter + face_text_locate
        answer = describles_answ[label]
        
        if has_bbox:
            ## florence2返回的坐标是xyxy格式的 x1,y1,x2,y2 = 365.4,465.2,765.8,999.6
            x1,y1,x2,y2 = self.denormalize_fake_image_box_xyxy(fake_image_box,W,H)
            # 保留两位小数，并插入到字符串模板中
            face_bbox_answer = (
                "Manipulated face"
                + f"<loc_{int(x1)}>"
                + f"<loc_{int(y1)}>"
                + f"<loc_{int(x2)}>"
                + f"<loc_{int(y2)}>"
                + '.'
            )
            answer += face_bbox_answer
        if len(fake_text_pos)>0:
            answer += self.extract_swapped_words(caption,fake_text_pos)
            
            ## 图像，问题，答案，fake word的[01]编码
        return image, question, answer,fake_text_pos_list,caption,fake_image_box

###########################


###########################


class APIinferDataset(Dataset):
    '''用于API推理的dataset类
    '''

    def __init__(self, split, data, max_words=30, image_res=224):
        self.name = "DGM4"
        
        self.data = data
        # self.transform = transform
        self.max_words = max_words
        self.image_res = image_res

        is_train = False
        if split == 'train':
            is_train = True
        
        self.is_train = is_train # 如果是is_train，那么会在训练集中随机加入flip增强
        self.task_prompt = "<VQA>"
        

    def __len__(self):
        return len(self.data)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)
    
    def denormalize_fake_image_box_xyxy(self, fake_image_box, image_width, image_height):
        """
        Converts a normalized fake_image_box in [center_x, center_y, w, h] format 
        to absolute xyxy coordinates with two decimal precision.
        
        Args:
            fake_image_box (torch.Tensor): Tensor containing normalized [center_x, center_y, w, h].
            image_width (int): The width of the original image.
            image_height (int): The height of the original image.
        
        Returns:
            tuple: (x1, y1, x2, y2) in absolute coordinates, rounded to two decimal places.
        """
        # Unpack the normalized coordinates
        center_x, center_y, w, h = fake_image_box

        # Convert normalized coordinates to absolute values
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_w = w * image_width
        abs_h = h * image_height

        # Calculate xyxy format
        x1 = abs_center_x - abs_w / 2
        y1 = abs_center_y - abs_h / 2
        x2 = abs_center_x + abs_w / 2
        y2 = abs_center_y + abs_h / 2

        # Round to two decimal places and return as a tuple
        return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)

    
    def extract_swapped_words(self, caption, fake_text_pos):
        """
        根据 fake_text_pos 提取 caption 中的 swapped words，并返回格式化字符串。

        参数：
        caption (str): 新闻的标题文本。
        fake_text_pos (list): 一个包含被篡改单词索引的列表（从0开始）。

        返回：
        str: 以 "Swapped words: xxx, xxx, xxx" 格式返回被篡改单词字符串。
        """
        words = caption.split()  # 按空格拆分 caption 成单词列表
        swapped_words = [words[i] for i in fake_text_pos if 0 <= i < len(words)]  # 提取被篡改的单词
        return f" Swapped words: {', '.join(swapped_words)}" if swapped_words else "Swapped words: None"


    
    def __getitem__(self, index):

        ann = self.data[index]
        label = ann['fake_cls'].replace('text_attribute','text_swap')

        
        img_dir = os.path.join('/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/',ann['image'])

        image_dir_all = img_dir
        
        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        mask = np.zeros((self.image_res,self.image_res,1))
        
        if any(keyword in label for keyword in ['face_swap', 'face_attribute']):
            try:
                x, y, w, h = self.get_bbox(ann['fake_image_box'])
                has_bbox = True
            except Exception:
                fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        else:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
                

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        # image = self.transform(image)

        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)
            
            mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, :] = 1

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                                           center_y / self.image_res,
                                           w / self.image_res,
                                           h / self.image_res],
                                          dtype=torch.float)


        caption = pre_caption(ann['text'], self.max_words)
        # 原始代码
        # fake_text_pos = ann['fake_text_pos']
        # 修改后 如果获取不到fake_text_pos，则fake_text_pos的值为[]，
        fake_text_pos = ann.get('fake_text_pos', [])


        fake_text_pos_list = torch.zeros(self.max_words)
        mask = torch.tensor(mask[None, ..., 0]).float()

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1


        # conversation = []
        # conversation.append(
        #     {"from": "human", "value": describe_temple + caption + describe_ques_latter})
        # conversation.append({"from": "gpt", "value": describles_answ[label]})
        
        question = '<DGM4>'+describe_temple + caption + describe_ques_latter + face_text_locate
        answer = describles_answ[label]
        
        if has_bbox:
            ## florence2返回的坐标是xyxy格式的 x1,y1,x2,y2 = 365.4,465.2,765.8,999.6
            x1,y1,x2,y2 = self.denormalize_fake_image_box_xyxy(fake_image_box,W,H)
            # 保留两位小数，并插入到字符串模板中
            face_bbox_answer = (
                "Manipulated face"
                + f"<loc_{int(x1)}>"
                + f"<loc_{int(y1)}>"
                + f"<loc_{int(x2)}>"
                + f"<loc_{int(y2)}>"
                + '.'
            )
            answer += face_bbox_answer
        if len(fake_text_pos)>0:
            answer += self.extract_swapped_words(caption,fake_text_pos)
            
            ## 图像，问题，答案，fake word的[01]编码
        if 'qwen-vl-max_output' in ann:
            API_answer = ann['qwen-vl-max_output']
        elif 'gpt-4o_output' in ann:
            API_answer = ann['gpt-4o_output']
        elif 'qwen3-235b-a22b_output' in ann:
            API_answer = ann['qwen3-235b-a22b_output']
        elif 'gemini-2.0-flash_output' in ann:
            API_answer = ann['gemini-2.0-flash_output']
        
        def API_answer_parse_coordinates(text,W,H):
            '''
            解析API返回的坐标,只解析并返回第一个坐标
            解析坐标的文本格式是 [x1, y1, x2, y2]
            坐标是相对坐标，还需要转换成绝对坐标
            '''
            pattern = r'\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]'
            match = re.search(pattern, text)
            if match:
                rel = [float(match.group(i)) for i in range(1, 5)]
                abs_coords = [
                        rel[0] * W,
                        rel[1] * H,
                        rel[2] * W,
                        rel[3] * H,
                    ]
                return torch.tensor([abs_coords])
            else:
                # return torch.tensor([[0, 0, 0, 0]])
                return torch.tensor([[-1, -1, -1, -1]])
            
        API_box = API_answer_parse_coordinates(API_answer,W,H)

        
        return image, question, answer,fake_image_box, API_answer, API_box

###########################


class OriDGM4DatasetOB(Dataset):
    '''用于原始的DGM4数据集的训练
        即包含的answer中有Fake words
    '''

    def __init__(self, split, data, max_words=30, image_res=224):
        self.name = "DGM4"
        
        self.data = data
        # self.transform = transform
        self.max_words = max_words
        self.image_res = image_res

        is_train = False
        if split == 'train':
            is_train = True
        
        self.is_train = is_train # 如果是is_train，那么会在训练集中随机加入flip增强
        self.task_prompt = "<VQA>"
        

    def __len__(self):
        return len(self.data)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)
    
    def denormalize_fake_image_box_xyxy(self, fake_image_box, image_width, image_height):
        """
        Converts a normalized fake_image_box in [center_x, center_y, w, h] format 
        to absolute xyxy coordinates with two decimal precision.
        
        Args:
            fake_image_box (torch.Tensor): Tensor containing normalized [center_x, center_y, w, h].
            image_width (int): The width of the original image.
            image_height (int): The height of the original image.
        
        Returns:
            tuple: (x1, y1, x2, y2) in absolute coordinates, rounded to two decimal places.
        """
        # Unpack the normalized coordinates
        center_x, center_y, w, h = fake_image_box

        # Convert normalized coordinates to absolute values
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_w = w * image_width
        abs_h = h * image_height

        # Calculate xyxy format
        x1 = abs_center_x - abs_w / 2
        y1 = abs_center_y - abs_h / 2
        x2 = abs_center_x + abs_w / 2
        y2 = abs_center_y + abs_h / 2

        # Round to two decimal places and return as a tuple
        return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)

    
    def extract_swapped_words(self, caption, fake_text_pos):
        """
        根据 fake_text_pos 提取 caption 中的 swapped words，并返回格式化字符串。

        参数：
        caption (str): 新闻的标题文本。
        fake_text_pos (list): 一个包含被篡改单词索引的列表（从0开始）。

        返回：
        str: 以 "Swapped words: xxx, xxx, xxx" 格式返回被篡改单词字符串。
        """
        words = caption.split()  # 按空格拆分 caption 成单词列表
        swapped_words = [words[i] for i in fake_text_pos if 0 <= i < len(words)]  # 提取被篡改的单词
        return f" Swapped words: {', '.join(swapped_words)}" if swapped_words else "Swapped words: None"


    
    def __getitem__(self, index):

        ann = self.data[index]
        label = ann['fake_cls'].replace('text_attribute','text_swap')

        
        img_dir = os.path.join('/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/',ann['image'])

        image_dir_all = img_dir
        
        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        mask = np.zeros((self.image_res,self.image_res,1))
        
        if any(keyword in label for keyword in ['face_swap', 'face_attribute']):
            try:
                x, y, w, h = self.get_bbox(ann['fake_image_box'])
                has_bbox = True
            except Exception:
                fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        else:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
                

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        # image = self.transform(image)

        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)
            
            mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w, :] = 1

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                                           center_y / self.image_res,
                                           w / self.image_res,
                                           h / self.image_res],
                                          dtype=torch.float)


        caption = pre_caption(ann['text'], self.max_words)
        # 原始代码
        # fake_text_pos = ann['fake_text_pos']
        # 修改后 如果获取不到fake_text_pos，则fake_text_pos的值为[]，
        fake_text_pos = ann.get('fake_text_pos', [])


        fake_text_pos_list = torch.zeros(self.max_words)
        mask = torch.tensor(mask[None, ..., 0]).float()

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1

        question = '<DGM4>'+describe_temple + caption + describe_ques_latter_OB
        
        answer = 'B. Yes.'
        if label == 'orig':
            answer = 'A. No.'
        
            ## 图像，问题，答案，fake word的[01]编码
        return image, question, answer,fake_text_pos_list,caption,fake_image_box



class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.name = "DocVQA"
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<VQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = first_answer
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image

    
class VQAInstructDataset(BaseDataset):
    def __init__(self, split, max_length=1024):
        super().__init__(split)
        self.name = "VQA-Instruct"
        self._max_length = max_length
        self.vqa_data = load_from_disk("HuggingFaceM4/Docmatix_single_images")
        split_actions = {
                'train': lambda data: data.train_test_split(test_size=0.05, seed=42)['train'],
                'validation': lambda data: data.train_test_split(test_size=0.05, seed=42)['test'].train_test_split(test_size=0.5, seed=42)['train'],
                'test': lambda data: data.train_test_split(test_size=0.05, seed=42)['test'].train_test_split(test_size=0.5, seed=42)['test']
            }

        if split not in split_actions:
            raise ValueError(f"Unknown split: {split}")

        self.vqa_data = split_actions[split](self.vqa_data)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.vqa_data)
    
    def __getitem__(self, idx):
        example = self.vqa_data[idx]
        texts = random.choice(example['texts'])

        question = self.task_prompt + texts["user"]
        answer = texts["assistant"]

        image = example['images']
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image

class TheCauldronDataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.name = "The-Cauldron"
        self.images_df, self.texts_df = self.load_all_configs(split)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.texts_df)
    
    def load_config(self, config_name, split):
        print(f"Loading config: {config_name}")
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config_name, split=split)
        print(f"Finished loading config: {config_name}")

        df_data = dataset.to_pandas()

        # Create the images DataFrame
        df_images = df_data[['images']].copy()
        df_images['image_index'] = df_images.index

        # Explode the texts into separate rows and create a DataFrame
        df_texts = df_data[['texts']].explode('texts').reset_index()
        df_texts.rename(columns={'index': 'image_index'}, inplace=True)

        # Extract 'user', 'assistant', and 'source' from the 'texts' column
        df_texts['question'] = df_texts['texts'].apply(lambda x: x.get('user'))
        df_texts['answer'] = df_texts['texts'].apply(lambda x: x.get('assistant'))
        df_texts['source'] = df_texts['texts'].apply(lambda x: x.get('source'))

        # Drop the original 'texts' column
        df_texts.drop(columns=['texts'], inplace=True)

        # Copy the 'source' column to the images df, using the first source per image index
        df_images = df_images.merge(df_texts[['image_index', 'source']], on='image_index', how='left')
        print(f"Finished processing config: {config_name}")

        return df_images, df_texts

    def load_all_configs(self, split):
        cauldron_config_names = get_dataset_config_names("HuggingFaceM4/the_cauldron")

        images_dfs = []
        texts_dfs = []

        # Use ThreadPoolExecutor for parallel processing and tqdm for progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  # Limit the number of workers
            with tqdm(total=len(cauldron_config_names), desc="Total Progress") as total_pbar:
                futures = {executor.submit(self.load_config, config_name, split): config_name for config_name in cauldron_config_names}
                for future in concurrent.futures.as_completed(futures):
                    config_name = futures[future]
                    try:
                        df_images, df_texts = future.result()
                        images_dfs.append(df_images)
                        texts_dfs.append(df_texts)
                    except Exception as exc:
                        print(f"{config_name} generated an exception: {exc}")
                    total_pbar.update(1)

        # Merge all the loaded DataFrames
        print("Merging DataFrames...")
        merged_images_df = pd.concat(images_dfs, ignore_index=True)
        merged_texts_df = pd.concat(texts_dfs, ignore_index=True)
        print("Finished merging DataFrames")

        return merged_images_df, merged_texts_df

    def __getitem__(self, idx):
        example = self.texts_df.iloc[idx]
        question = example["question"]
        answer = example["answer"]
        source = example["source"]
        image_idx = example["image_index"]

        image_data = self.images_df.loc[(self.images_df['image_index'] == image_idx) & (self.images_df['source'] == source), 'images'].values[0][0]['bytes'] 
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image
