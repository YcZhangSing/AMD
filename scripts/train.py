import argparse
import os
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
import json
import re
from functools import partial
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
from data import DocVQADataset, TheCauldronDataset, VQAInstructDataset,DGM4_Dataset
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from multilabel_metrics import AveragePrecisionMeter
from torchvision.ops.boxes import box_area



def set_seed(seed, rank=0):

    seed = seed + rank  
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
    image_sizes = [(image.width, image.height) for image in images]
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers,fake_image_box,image_sizes


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
    
    # face_swap&text_swap cls = [1, 0, 1, 0]
    pos = [
        i for i, ans in enumerate(answers)
        if ('E. Face swap and text swap.' in ans) or ('E. Both face swap and text swap.' in ans)
    ]
    multi_label[pos, :] = torch.tensor([1, 0, 1, 0]).to(device)
    
    # face_attribute&text_swap cls = [0, 1, 1, 0]
    pos = [
        i for i, ans in enumerate(answers)
        if ('F. Face attribute and text swap.' in ans) or ('F. Both face attribute and text swap.' in ans)
    ]
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


def parse_coordinates(text, image_size):
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    match = re.search(pattern, text)
    
    if match:
        image_width, image_height = image_size
        loc_x1 = max(0, min(int(match.group(1)), 999))
        loc_y1 = max(0, min(int(match.group(2)), 999))
        loc_x2 = max(0, min(int(match.group(3)), 999))
        loc_y2 = max(0, min(int(match.group(4)), 999))
        x1 = (loc_x1 + 0.5) * image_width / 1000
        y1 = (loc_y1 + 0.5) * image_height / 1000
        x2 = (loc_x2 + 0.5) * image_width / 1000
        y2 = (loc_y2 + 0.5) * image_height / 1000
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    else:
        return torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)


def denormalize_fake_image_box_xyxy(fake_image_box, image_size):
    image_width, image_height = image_size
    box = torch.as_tensor(fake_image_box, dtype=torch.float32)
    if box.numel() < 4 or torch.all(torch.abs(box[:4]) < 1e-6):
        return torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
    center_x, center_y, width, height = box[:4]
    abs_center_x = center_x * image_width
    abs_center_y = center_y * image_height
    abs_width = width * image_width
    abs_height = height * image_height
    x1 = abs_center_x - abs_width / 2
    y1 = abs_center_y - abs_height / 2
    x2 = abs_center_x + abs_width / 2
    y2 = abs_center_y + abs_height / 2
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


def evaluate_model(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count, criterion, regular_weight):

    model.eval()
    with torch.inference_mode():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            # total, llm, image, text, learnable token, bbox, giou, regular
            local_loss_sums = torch.zeros(8, dtype=torch.float32, device=device)
            local_batch_count = torch.tensor(0.0, dtype=torch.float32, device=device)

            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                inputs, answers, fake_image_box, _ = batch
                val_item_count += len(answers)

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

                binary_labels = torch.tensor(
                    [1 if label.startswith("A") else 0 for label in answers],
                    dtype=torch.long,
                    device=device,
                )

                image_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                text_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                learnable_token_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                bbox_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                giou_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                regular_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                logits_list = outputs.classification_logits_list
                for i, logits in enumerate(logits_list):
                    if logits is None:
                        continue
                    if i == 0:
                        image_loss = criterion(logits, binary_labels)
                        total_loss += 0.1 * image_loss
                    if i == 1:
                        text_loss = criterion(logits, binary_labels)
                        total_loss += 0.1 * text_loss
                    if i == 2:
                        learnable_token_loss = criterion(logits, binary_labels)
                        total_loss += 0.1 * learnable_token_loss
                    if i == 3:
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        bbox_loss, giou_loss = get_bbox_loss(output_coords, tensor_fake_image_box)
                        total_loss += 0.1 * (bbox_loss + giou_loss)
                    if i == 4:
                        regular_loss = regular_weight * logits.to(device)
                        total_loss += regular_loss

                local_loss_sums += torch.stack(
                    [
                        total_loss.detach(),
                        outputs.loss.detach(),
                        image_loss.detach(),
                        text_loss.detach(),
                        learnable_token_loss.detach(),
                        bbox_loss.detach(),
                        giou_loss.detach(),
                        regular_loss.detach(),
                    ]
                )
                local_batch_count += 1

                if val_item_count >= max_val_item_count:
                    break

            dist.reduce(local_loss_sums, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(local_batch_count, dst=0, op=dist.ReduceOp.SUM)

            if dist.get_rank() == 0 and local_batch_count.item() > 0:
                avg_loss = local_loss_sums / local_batch_count
                wandb.log(
                    {
                        "step": global_step,
                        f"{val_name}_val_total_loss": avg_loss[0].item(),
                        f"{val_name}_val_llm_loss": avg_loss[1].item(),
                        f"{val_name}_val_image_loss": avg_loss[2].item(),
                        f"{val_name}_val_text_loss": avg_loss[3].item(),
                        f"{val_name}_val_learnable_token_loss": avg_loss[4].item(),
                        f"{val_name}_val_bbox_loss": avg_loss[5].item(),
                        f"{val_name}_val_giou_loss": avg_loss[6].item(),
                        f"{val_name}_val_regular_loss": avg_loss[7].item(),
                    }
                )

    model.train()



def train_model(rank, AMD_init_pth, train_js, val_js, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000, regular_weight=0.07, train_domain='NYT',random_seed=12):
    setup(rank, world_size)
    set_seed(random_seed, rank)
    device = torch.device(f"cuda:{rank}")
    train_data=[]
    val_data=[]
    
    logged_task_name = f'AMD_test_{train_domain}' 
    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_train_name = f'{logged_task_name}_{train_time}' 
    
    criterion = torch.nn.CrossEntropyLoss() 
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Persist experiment settings under the current run directory (same level as epoch_* checkpoints).
    out_put_prefix = './AMD_log'
    run_root_dir = os.path.join(out_put_prefix, f"train_{train_time}")
    if rank == 0:
        os.makedirs(run_root_dir, exist_ok=True)
        exp_settings_path = os.path.join(run_root_dir, "exp_settings.log")
        with open(exp_settings_path, "a", encoding="utf-8") as f:
            f.write("===== Training Launch =====\n")
            f.write(f"start_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"run_name={run_name}\n")
            f.write(f"logged_task_name={logged_task_name}\n")
            f.write(f"dataset={dataset_name}\n")
            f.write(f"train_domain={train_domain}\n")
            f.write(f"AMD_init_pth={AMD_init_pth}\n")
            f.write(f"train_js={train_js}\n")
            f.write(f"val_js={val_js}\n")
            f.write(f"batch_size={batch_size}\n")
            f.write(f"epochs={epochs}\n")
            f.write(f"lr={lr}\n")
            f.write(f"eval_steps={eval_steps}\n")
            f.write(f"max_val_item_count={max_val_item_count}\n")
            f.write(f"regular_weight={regular_weight}\n")
            f.write(f"random_seed={random_seed}\n")
            f.write(f"world_size={world_size}\n")
            f.write(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
            f.write("===========================\n")

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
        with open(train_js, "r") as f:
            train_data = json.load(f)
        with open(val_js, "r") as f:
            val_data = json.load(f)
            
        train_dataset = DGM4_Dataset(split='train',data=train_data)
        val_datasets = {"DGM4": DGM4_Dataset(split='validation',data=val_data)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        AMD_init_pth, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        AMD_init_pth, trust_remote_code=True
    )
    

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
        loss_list = []
        image_loss = 0
        text_loss = 0
        LT_loss = 0
        loss_regular = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers,fake_image_box,image_sizes = batch

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
            Binary_lables = []
            for tt, label in enumerate(answers):
                if label.startswith('A'):
                    Binary_lables.append(1)  
                else:
                    Binary_lables.append(0)  
                    
            Binary_lables = torch.tensor(Binary_lables, dtype=torch.long).to(device)

            logits_list = outputs.classification_logits_list
            ### logits = [image_classification, text_classification,learnable_token_logits,output_coord,loss_regular]
            
            for i,logits in enumerate(logits_list):
                if logits is not None:
                    if i == 0:
                        temp_loss0 = criterion(logits,Binary_lables) 
                        if torch.isnan(temp_loss0):
                            raise RuntimeError(f"❌ logits_list[{i}] 产生 NaN，二分类损失 temp_loss0 为 NaN")
                        total_loss += 0.1*temp_loss0 
                        loss_list.append(temp_loss0)
                    if i == 1:
                        temp_loss1 = criterion(logits,Binary_lables) 
                        if torch.isnan(temp_loss1):
                            raise RuntimeError(f"❌ logits_list[{i}]  temp_loss1 = NaN")
                        total_loss += 0.1*temp_loss1 
                        loss_list.append(temp_loss1)
                    if i == 2:
                        temp_loss2 = criterion(logits,Binary_lables) 
                        if torch.isnan(temp_loss2):
                            raise RuntimeError(f"❌ logits_list[{i}]  temp_loss2 = NaN")
                        total_loss += 0.1*temp_loss2 
                        loss_list.append(temp_loss2)
                        
                    if i == 3: ##utput_coord
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        loss_bbox, loss_giou = get_bbox_loss(output_coords, tensor_fake_image_box) 
                        if torch.isnan(loss_bbox):
                            raise RuntimeError(f"❌ logits_list[{i}] loss_bbox = NaN")
                        total_loss += 0.1*(loss_bbox+loss_giou) 
                        loss_list.append(loss_bbox)
                        loss_list.append(loss_giou)
                    
                    if i == 4: 
                        loss_regular = logits.to(device)
                        if torch.isnan(loss_regular):
                            raise RuntimeError(f"❌ logits_list[{i}] loss_regular = NaN")
                        loss_regular = regular_weight * loss_regular
                        loss_list.append(loss_regular)
                        total_loss += loss_regular

    
            total_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += total_loss.item()
            LLM_loss += outputs.loss.item()
            image_loss += loss_list[0].item()
            text_loss += loss_list[1].item()
            LT_loss += loss_list[2].item()
            loss_bbox += loss_list[3].item()
            loss_giou += loss_list[4].item()
            loss_regular += loss_list[5].item()
            
            
            
            if rank == 0:
                wandb.log(
                    {
                        "step": global_step + 1,
                        "step_train_loss": total_loss.item(),
                        "step_avg_LLM_loss": outputs.loss.item(),
                        "step_avg_image_loss": loss_list[0].item(),
                        "step_avg_text_loss": loss_list[1].item(),
                        "step_avg_LearnableToken_loss": loss_list[2].item(),
                        "step_avg_bbox_loss": loss_list[3].item(),
                        "step_avg_giou_loss": loss_list[4].item(),
                        "step_avg_regular_loss": loss_list[5].item(),
                    }
                )
                
            loss_list.clear()    
            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count, criterion, regular_weight)

        evaluate_model(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count, criterion, regular_weight)

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
            wandb.log(
                {
                    "step": global_step,
                    "epoch_train_loss": avg_train_loss,
                    "epoch_avg_LLM_loss": avg_LLM_loss,
                    "epoch_avg_image_loss": avg_image_loss,
                    "epoch_avg_text_loss": avg_text_loss,
                    "epoch_avg_LearnableToken_loss": avg_LT_loss,
                    "epoch_avg_bbox_loss": avg_bbox_loss,
                    "epoch_avg_giou_loss": avg_giou_loss,
                    "epoch_avg_regular_loss": avg_regular_loss,
                }
            )


        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            output_dir = os.path.join(run_root_dir, f"epoch_{epoch+1}")
            
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train AMD model on specified dataset")
    parser.add_argument("--AMD-init-pth", type=str, help="AMD model dir")
    parser.add_argument("--dataset-type", type=str, default="DGM4", choices=["docvqa", "cauldron", "vqainstruct","DGM4"], help="Dataset to train on")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for training") 
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=13, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=2000, help="Number of steps between evaluations") 
    parser.add_argument("--run-name", type=str, default='test', help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=2000, help="Maximum number of items to evaluate on during validation")
    parser.add_argument("--regular-weight", type=float, default=0.05, help="loss weight of L_TRP")
    parser.add_argument("--train-js", type=str, default='./train.json', help="json file for train")
    parser.add_argument("--val-js", type=str, default='./val.json', help="json file for val")
    parser.add_argument("--train-domain", type=str, default='NYT', help="News domain of train data")
    parser.add_argument("--seed", type=int, default=12, help="random seed, small is better")
    
    
    
    
    
    
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(args.AMD_init_pth, args.train_js, args.val_js, world_size, args.dataset_type, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count, args.regular_weight, args.train_domain),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    main()
