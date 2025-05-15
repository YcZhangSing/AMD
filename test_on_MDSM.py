import logging
import random
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from data import DGM4_Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys, re
from multilabel_metrics import AveragePrecisionMeter
import os,math

import datetime
import time


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

GPU_nu = 0
device = torch.device(f"cuda:{GPU_nu}" if torch.cuda.is_available() else "cpu")
batch_size = 6  

model_id = '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/florence2-finetuning/zyc_ft_log/train_20250421_201644_MDSM_guardian_GCN_0P1regu/guardian_GCN_0P1regu_epoch_10' 

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)



output_file = os.path.join(model_id, f"domain_test_out_{timestamp}.txt")
# 先定义 log_print 函数，使其可用于整个脚本
def log_print(*args, **kwargs):
    print(*args, **kwargs)  # 控制台输出
    with open(output_file, "a") as flog:  # 以追加模式写入文件
        print(*args, **kwargs, file=flog)


vals = [
    '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/media_source_data/bbc.json',
    '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/media_source_data/usa_today.json',
    '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/media_source_data/washington_post.json',  
    '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/NYT_dataset/meta_data/test_fromNYT.json',
    '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/guardian_dataset/meta_data/test.json',     
]


options = [
    "A. No.",
    "B. Only face swap.",
    "C. Only face attribute.",
    "D. Only text swap.",
    "E. Face swap and text swap.",
    "F. Face attribute and text swap.",
]
vectorizer = TfidfVectorizer().fit(options)
option_vectors = vectorizer.transform(options).toarray()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create DataLoader
GPU_nu = torch.cuda.device_count()
num_workers = 0  # Number of worker processes to use for data loading
prefetch_factor = None  # Number of batches to prefetch



def get_best_option(generated_texts, option_vectors):
    '''批量计算模型的输出对应哪一个选项
    输入是生成的多个文本，和固定选项的向量表示
    '''
    # 将生成文本批量转换为向量
    generated_vectors = vectorizer.transform(generated_texts).toarray()

    # 计算相似度
    similarities = cosine_similarity(generated_vectors, option_vectors)

    # 获取每个生成文本的相似度最高的选项
    best_option_indices = similarities.argmax(axis=1)

    # 返回选项、相似度和对应的01标签
    best_options = [options[i] for i in best_option_indices]
    best_similarities = [similarities[i, best_option_indices[i]] for i in range(len(generated_texts))]

    best_multi_labels = torch.stack([option_labels[i] for i in best_option_indices], dim=0)
    # 对 best_multi_labels 进行归一化
    # best_multi_labels_prob = F.softmax(best_multi_labels.float(), dim=1)
    
    #ori_pos，构造模型输出对应的单分类标签
    pred_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
    real_label_pos = np.where(np.array(best_options) == 'A. No.')[0].tolist()
    # 是A. No.的地方设置为 0 --代表real图文
    pred_label[real_label_pos] = 0
    
    return best_options, best_similarities, best_multi_labels,pred_label

def get_multi_label(answers):
    # 初始化 multi_label 矩阵
    multi_label = torch.zeros([len(answers), 4], dtype=torch.long).to(device)
    
    # 定义 real_label_pos（精确匹配 'A. No.'）
    real_label_pos = [i for i, ans in enumerate(answers) if 'A. No.' in ans ]
    multi_label[real_label_pos, :] = torch.tensor([0, 0, 0, 0]).to(device)
    
    # face_swap cls = [1, 0, 0, 0]（精确匹配 'B. Only face swap.'）
    pos = [i for i, ans in enumerate(answers) if 'B. Only face swap.' in ans ]
    multi_label[pos, :] = torch.tensor([1, 0, 0, 0]).to(device)
    
    # face_attribute cls = [0, 1, 0, 0]（精确匹配 'C. Only face attribute.'）
    pos = [i for i, ans in enumerate(answers) if 'C. Only face attribute.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 1, 0, 0]).to(device)
    
    # text_swap cls = [0, 0, 1, 0]（精确匹配 'D. Only text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'D. Only text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 0, 1, 0]).to(device)
    
    # face_swap&text_swap cls = [1, 0, 1, 0]（精确匹配 'E. Face swap and text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'E. Face swap and text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([1, 0, 1, 0]).to(device)
    
    # face_attribute&text_swap cls = [0, 1, 1, 0]（精确匹配 'F. Face attribute and text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'F. Face attribute and text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 1, 1, 0]).to(device)
    
    return multi_label, real_label_pos


# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer


def collate_fn(batch):
    # questions, answers, images = zip(*batch)
    images, questions, answers,fake_image_box = zip(*batch)
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers


def run_batch(inputs):
    # 调用 modeling_florence2.py 中的generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts


option_labels = [
    torch.tensor([0, 0, 0, 0]).to(device),
    torch.tensor([1, -0.33, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, 1, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, -0.33, 1, -0.33]).to(device),
    torch.tensor([0.5, -0.5, 0.5, -0.5]).to(device),
    torch.tensor([-0.5, 0.5, 0.5, -0.5]).to(device),
]


def box_iou(boxes1, boxes2, test=False):
    '''
    计算两个边界框集合的 IoU（Intersection over Union），
    并返回每个边界框对的 IoU 值和并集面积。
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    # union = area1[:, None] + area2 - inter
    union = area1 + area2 - inter

    iou = inter / union

    if test:
        zero_lines = boxes2==torch.zeros_like(boxes2)
        zero_lines_idx = torch.where(zero_lines[:,0]==True)[0]

        for idx in zero_lines_idx:
            if all(boxes1[idx,:] < 1e-4):
                iou[idx]=1

    return iou, union

def parse_coordinates(text):
    # 使用正则表达式匹配坐标
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    match = re.search(pattern, text)
    # print(f'input text is {text}')
    
    if match:
        # 将匹配到的坐标转换为整数
        loc_x1 = int(match.group(1))
        loc_y1 = int(match.group(2))
        loc_x2 = int(match.group(3))
        loc_y2 = int(match.group(4))
        # print('解析到的坐标是：')
        # print(loc_x1, loc_y1, loc_x2, loc_y2)
        return torch.tensor([[loc_x1, loc_y1, loc_x2, loc_y2]])
    else:
        # print('没有match')
        return torch.tensor([[0, 0, 0, 0]])

def evaluate_model(test_loader):

    IOU_pred = []
    cls_nums_all = 0
    cls_acc_all = 0   
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()


    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        
        generated_texts = run_batch(inputs)
        task_answers = []
        output_coords = torch.zeros((len(generated_texts), 4)).to(device)
        true_coords = torch.zeros((len(generated_texts), 4)).to(device)
        
        
        for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):

            full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
            
            if '<loc_' in full_answer:
                task_answers.append(full_answer.split('Manipulated face')[0])
                output_coords[i] = parse_coordinates(full_answer).to(device)
                true_coords[i] = parse_coordinates(answers).to(device)
            # 将 output_coord 堆叠到 output_coords中
            else:
                task_answers.append(full_answer)
                true_coords[i] = parse_coordinates(answers).to(device)
                
        
                
        real_multi_label, real_label_pos = get_multi_label(batch_answers)
        real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
        real_label[real_label_pos] = 0
        best_options, _ ,best_multi_labels,pred_label = get_best_option(task_answers, option_vectors)
        IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)

        # IOU_pred.extend(IOU.cpu().tolist())
        # 遍历 IOU 并检查是否为有效数字--改进，当IOU是有效数字，才会加入IOU_Pred
        for iou_value in IOU.cpu().tolist():
            if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                IOU_pred.append(iou_value)
            else:
                IOU_pred.append(0.0)
        ######################################
        ##--reeal/fake---##
        cls_nums_all += len(generated_texts)
        cls_acc_all += torch.sum(real_label == pred_label).item()
        ##-multi--##
        multi_label_meter.add(best_multi_labels, real_multi_label)
        

    IOU_score = sum(IOU_pred)/len(IOU_pred)
    
    
    ACC_cls = cls_acc_all / cls_nums_all
    
    MAP = multi_label_meter.value()[:3].mean()
    
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()

    return ACC_cls, cls_acc_all, cls_nums_all, MAP,OP, OR, OF1, CP, CR, CF1,IOU_score,IOU_pred

        
log_print(f'本次测试使用的权值 modelid_is {model_id},测试集使用比例是{test_rat}')  


for val_js in vals:
    with open(val_js, "r") as f:
        val_data = json.load(f)
    log_print(f'测试集长度为{len(val_data)}')
    test_dataset = DGM4_Dataset(split="validation",data=val_data)
    
    test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
)

    # Run the evaluation
    ACC_cls, cls_acc_all, cls_nums_all, MAP,OP, OR, OF1, CP, CR, CF1,IOU_score,IOU_pred = evaluate_model(test_loader)

    log_print('#######<--record-->###########')
    log_print(f'此测试集：{val_js} 的性能如下：')
    log_print(f"ACC_cls (Accuracy): {ACC_cls*100} (cls_acc_all: {cls_acc_all}, cls_nums_all: {cls_nums_all})")
    log_print(f"MAP (Mean Average Precision): {MAP*100}")
    log_print(f"IoUscore: {IOU_score*100}")
    log_print('########<--record-->#########')
    log_print(f"Overall Precision (OP): {OP}")
    log_print(f"Overall Recall (OR): {OR}")
    log_print(f"Overall F1 (OF1): {OF1}")
    log_print(f"Class Precision (CP): {CP}")
    log_print(f"Class Recall (CR): {CR}")
    log_print(f"Class F1 (CF1): {CF1}")
    log_print("END############################################################################################")

print(f"日志已保存到 {output_file}")
