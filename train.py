import argparse
import os
import json
import re
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from datetime import datetime
import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
import torch.nn.functional as F
import box_ops
import math

import random
import wandb
## 使用wandb来管理模型训练日志
## wandb: Currently logged in as: zycddl (yczhang). Use `wandb login --relogin` to force relogin
from data import DocVQADataset, TheCauldronDataset, VQAInstructDataset,DGM4_Dataset
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from multilabel_metrics import AveragePrecisionMeter
from torchvision.ops.boxes import box_area


train_data = []
val_data = []
train_epoch = 11
train_batch = 5
regular_weight = 0.1
random_seed = 12 
os.environ['PYTHONHASHSEED'] = str(random_seed)

train_domain = 'gurd'
train_js = '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/guardian_dataset/meta_data/train.json'
val_js = '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/domain_dataset_44w/guardian_dataset/meta_data/val.json'
##<------不同训练域--------->


florence_init_pth = '/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/microsoft_Florence_2_ForDGM4_LT32_AMD_GCN_Parral_regular'
logged_task_name = f'florence2_MDSM_AMD_GNN_0.1+0.2regular{train_domain}' 
train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
local_train_name = f'{logged_task_name}_{train_time}' 

def set_seed(seed, rank=0):
    '''
    set_seed函数要在train函数中调用
    因为我们使用的是多卡训练，需要保证每个rank的随机种子都设置好
    '''
    seed = seed + rank  # 确保每个进程种子不同但可控
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_bbox_loss(output_coord, target_bbox, is_image=None):
    """
    Bounding Box Loss: L1 & GIoU

    Args:
        image_embeds: encoding full images
    """
    loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

    boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
    boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
    if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
        # early check of degenerated boxes
        # print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
        loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
    else:
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
        loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

    if is_image is None:
        num_boxes = target_bbox.size(0)
    else:
        num_boxes = torch.sum(1 - is_image)
        loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
        loss_giou = loss_giou * (1 - is_image)

    return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


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

def collate_fn(batch, processor, device):

    #### DGM4的定义：
    images, questions, answers,fake_image_box = zip(*batch)
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers,fake_image_box


def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size//2,
            collate_fn=partial(collate_fn, processor=processor, device=device),
            num_workers=num_workers,
            sampler=val_sampler,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders

def get_multi_label(answers,device):
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




def get_best_option(generated_texts, option_vectors,vectorizer,options,option_labels,device):
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

def synchronize_metrics(metric_tensor, world_size):
    """
    使用reduce操作同步指标。聚合各个进程的指标，计算全局值。
    """
    # 将指标结果归约到 rank 0 进程
    dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)
    # 在 rank 0 进程计算均值
    if dist.get_rank() == 0:
        metric_tensor /= world_size
    return metric_tensor


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
        print('没有match')
        return torch.tensor([[0, 0, 0, 0]])



def evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels):

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            cls_nums_all = 0
            cls_acc_all = 0 
            IOU_pred = []
            multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
            multi_label_meter.reset()
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                inputs, batch_answers, fake_image_box = batch
                val_item_count += len(inputs)
                ## model是DistributedDataParallel包装后的类，并没有generate方法，如需调用，应该使用model.module调用基础模型后再调用generate()方法
                generated_ids = model.module.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                ###解析得到模型的文本输出
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
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
                
                
                real_multi_label, real_label_pos = get_multi_label(batch_answers,device)
                real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
                real_label[real_label_pos] = 0
                best_options, _ ,best_multi_labels,pred_label = get_best_option(task_answers, option_vectors,vectorizer,options,option_labels,device)
                
                ##--reeal/fake---##
                cls_nums_all = val_item_count
                cls_acc_all += torch.sum(real_label == pred_label).item()
                
                ##-IoU--##
                IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)
                
                # 遍历 IOU 并检查是否为有效数字--改进，当IOU是有效数字，才会加入IOU_Pred
                for iou_value in IOU.cpu().tolist():
                    if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                        IOU_pred.append(iou_value)
                    else:
                        IOU_pred.append(0.0)
                ######################################
                            
                ##-multi--##
                multi_label_meter.add(best_multi_labels, real_multi_label)
                
                # 计算本进程的各项指标
                local_ACC_cls = cls_acc_all / cls_nums_all
                local_IOU_score = sum(IOU_pred)/len(IOU_pred)
                local_MAP = multi_label_meter.value()[:3].mean().item()


                if val_item_count > max_val_item_count:
                    break
        # 同步各进程计算的指标
        local_ACC_cls_tensor = torch.tensor(local_ACC_cls, device=device)
        local_IoU_score_tensor = torch.tensor(local_IOU_score, device=device)
        local_MAP_tensor = torch.tensor(local_MAP, device=device)


        # 聚合指标到主进程
        ACC_cls = synchronize_metrics(local_ACC_cls_tensor, world_size)
        IoUscore = synchronize_metrics(local_IoU_score_tensor, world_size)
        MAP = synchronize_metrics(local_MAP_tensor, world_size)


        # 打印和记录日志
        if dist.get_rank() == 0:
            print(f"Rank {rank} - Step {global_step} - ACC perform ({val_name}): {ACC_cls.item()}")
            wandb.log({
                f"{val_name}_ACC_cls": ACC_cls.item(),
                f"{val_name}_IoUscore": IoUscore.item(),
                f"{val_name}_MAP": MAP.item(),
                "step": global_step
            })
            
    model.train()



def train_model(rank, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000):
    setup(rank, world_size)
    set_seed(random_seed, rank)
    device = torch.device(f"cuda:{rank}")
    print(f'使用这些  {device}  设备开始训练')
    
    # 实例化用于二分类损失计算的交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss() 
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    option_labels = [
    torch.tensor([0, 0, 0, 0]).to(device),
    torch.tensor([1, -0.33, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, 1, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, -0.33, 1, -0.33]).to(device),
    torch.tensor([0.5, -0.5, 0.5, -0.5]).to(device),
    torch.tensor([-0.5, 0.5, 0.5, -0.5]).to(device),
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
    
    # Initialize wandb
    if rank == 0:  # Only initialize wandb in the main process
        wandb.init(project= logged_task_name, name=run_name)
        wandb.config.update({
            "dataset": dataset_name,
            "batch_size": batch_size,
            "use_lora": use_lora,
            
            "epochs": epochs,
            "learning_rate": lr,
            "eval_steps": eval_steps,
            "world_size": world_size,
        })

    # Load the dataset based on the dataset_name argument
    ## 为了实现多个数据集支持，验证集采用键值对的形式
    if dataset_name == "docvqa":
        train_dataset = DocVQADataset(split='train')
        val_datasets = {"docvqa": DocVQADataset(split='validation')}
    elif dataset_name == "cauldron":
        train_dataset = TheCauldronDataset(split='train')
        val_datasets = {
            "cauldron": TheCauldronDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'vqainstruct':
        train_dataset = VQAInstructDataset(split='train')
        val_datasets = {
            "vqainstruct": VQAInstructDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'DGM4':
        # 加载训练集
        with open(train_js, "r") as f:
            train_data = json.load(f)
        # 加载验证集
        with open(val_js, "r") as f:
            val_data = json.load(f)
            
        train_dataset = DGM4_Dataset(split='train',data=train_data)
        val_datasets = {"DGM4": DGM4_Dataset(split='validation',data=val_data)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        florence_init_pth, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        florence_init_pth, trust_remote_code=True
    )
    
    #  # 计算参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print("<---------模型参数量计算------------>")
    # print(f"Total Parameters: {total_params:,}")  # 添加千分符便于阅读
    # print(f"Trainable Parameters: {trainable_params:,}")
    # print("<---------模型参数量计算------------>")

    if use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[rank])

    # Create DataLoaders
    num_workers = 0
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)  #lr=1e-6
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        LLM_loss = 0
        ##loss_list归零初始化
        loss_list = []
        image_loss = 0
        text_loss = 0
        LT_loss = 0
        loss_regular = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers,fake_image_box = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            total_loss = outputs.loss
            # 生成二分类标签
            Binary_lables = []
            for tt, label in enumerate(answers):
                if label.startswith('A'):
                    Binary_lables.append(1)  # 正样本
                else:
                    Binary_lables.append(0)  # 负样本
                    
            # 转换为 PyTorch 张量
            Binary_lables = torch.tensor(Binary_lables, dtype=torch.long).to(device)

            logits_list = outputs.classification_logits_list
            ### 这里loss_list列表的使用还可以优化，但是怕，如果不使用不同的变量名，会不会导致损失优化目标丢失？
            ### logits = [image_classification, text_classification,learnable_token_logits,output_coord,loss_regular]
            
            for i,logits in enumerate(logits_list):
                if logits is not None:
                    if i == 0:
                        temp_loss0 = criterion(logits,Binary_lables) # 计算二分类的二值交叉熵损失,image as query
                        if torch.isnan(temp_loss0):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，二分类损失 temp_loss0 为 NaN")
                        total_loss += 0.1*temp_loss0 ##给二分类加个权
                        loss_list.append(temp_loss0)
                    if i == 1:
                        temp_loss1 = criterion(logits,Binary_lables) # 计算二分类的二值交叉熵损失,image as query
                        if torch.isnan(temp_loss1):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，二分类损失 temp_loss1 为 NaN")
                        total_loss += 0.1*temp_loss1 ##给二分类加个权
                        loss_list.append(temp_loss1)
                    if i == 2:
                        temp_loss2 = criterion(logits,Binary_lables) # 计算二分类的二值交叉熵损失,image as query
                        if torch.isnan(temp_loss2):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，二分类损失 temp_loss2 为 NaN")
                        total_loss += 0.1*temp_loss2 ##给二分类加个权
                        loss_list.append(temp_loss2)
                        
                    if i == 3: ##这时是第四个返回，是output_coord
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        loss_bbox, loss_giou = get_bbox_loss(output_coords, tensor_fake_image_box) ## output_coords是归一化后的坐标
                        if torch.isnan(loss_bbox):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，坐标损失 loss_bbox 为 NaN")
                        total_loss += 0.1*(loss_bbox+loss_giou) ##给坐标损失加个权
                        loss_list.append(loss_bbox)
                        loss_list.append(loss_giou)
                    
                    if i == 4: ##这时是第五个返回，是loss_regular
                        loss_regular = logits.to(device)
                        if torch.isnan(loss_regular):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，regular损失 loss_regular 为 NaN")
                        loss_regular = regular_weight * loss_regular
                        loss_list.append(loss_regular)
                        total_loss += loss_regular
                        
                # else:
                #     print(f'Attention!! !!模型返回的logits[{i}]是None!!! ')
    
            total_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # #[image_loss,text_loss,LearnableToken loss] logits_list的顺序
            # # 消融时，记得调整下面的记录
            train_loss += total_loss.item()
            LLM_loss += outputs.loss.item()
            image_loss += loss_list[0].item()
            text_loss += loss_list[1].item()
            LT_loss += loss_list[2].item()
            loss_bbox += loss_list[3].item()
            loss_giou += loss_list[4].item()
            loss_regular += loss_list[5].item()
            
            
            
            if rank == 0:
                wandb.log({"step": global_step + 1, "step_train_loss": total_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_LLM_loss": outputs.loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_image_loss": loss_list[0].item()})
                wandb.log({"step": global_step + 1, "step_avg_text_loss": loss_list[1].item()})
                wandb.log({"step": global_step + 1, "step_avg_LearnableToken_loss": loss_list[2].item()})
                wandb.log({"step": global_step + 1, "step_avg_bbox_loss": loss_list[3].item()})
                wandb.log({"step": global_step + 1, "step_avg_giou_loss": loss_list[4].item()})
                wandb.log({"step": global_step + 1, "step_avg_regular_loss": loss_list[5].item()})
                
            loss_list.clear()    
            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels)

        evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels)

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        avg_LLM_loss = LLM_loss / len(train_loader)
        avg_image_loss = image_loss / len(train_loader)
        avg_text_loss = text_loss / len(train_loader)
        avg_LT_loss = LT_loss / len(train_loader)
        avg_bbox_loss = loss_bbox / len(train_loader)
        avg_giou_loss = loss_giou / len(train_loader)
        avg_regular_loss = loss_regular / len(train_loader)
    
        
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_LLM_loss": avg_LLM_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_image_loss": avg_image_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_text_loss": avg_text_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_LearnableToken_loss": avg_LT_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_bbox_loss": avg_bbox_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_giou_loss": avg_giou_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_regular_loss": avg_regular_loss})


        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            out_put_prefix = './AMD_log'
            output_dir = os.path.join(out_put_prefix,f"./train_{train_time}/epoch_{epoch+1}")
            
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        wandb.finish()

    cleanup()


def main(time_label):
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--dataset", type=str, default="DGM4", choices=["docvqa", "cauldron", "vqainstruct","DGM4"], help="Dataset to train on")
    parser.add_argument("--batch-size", type=int, default=train_batch, help="Batch size for training") 
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=train_epoch, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--eval-steps", type=int, default=2000, help="Number of steps between evaluations") 
    
    parser.add_argument("--run-name", type=str, default=local_train_name, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=2000, help="Maximum number of items to evaluate on during validation")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.dataset, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    main(train_time)