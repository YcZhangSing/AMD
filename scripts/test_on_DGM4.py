import logging
import argparse
import random
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from data import OriDGM4Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys, re
from multilabel_metrics import AveragePrecisionMeter
import os,math
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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


# Function to run the model on an example
def run_example(task_prompt, text_input, image,model, processor,device):
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



def run_batch(inputs,model, processor):
    # 调用 modeling_florence2.py 中的generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts




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

def compute_token_acc(captions, pre_words, fake_text_pos_list,tokenizer):
    """
    计算 token 级别的预测准确率（ACC）
    
    参数：
    captions: List[str]，存储完整句子，例如 ["I love transformer very much!"]
    pre_words: List[str]，存储逗号分隔的预测假单词，例如 ["transformer, very"]
    fake_text_pos_list: List[List[int]]，存储真实的假单词位置，例如 [[0, 0, 1, 0, 0]]
    tokenizer: 预训练 tokenizer（Hugging Face 兼容的 tokenizer）
    
    返回：
    float，表示 token 级别的预测准确率
    """
    assert len(captions) == len(pre_words) == len(fake_text_pos_list), "数据维度不匹配"

    TP, TN, FP, FN = 0, 0, 0, 0  # 统计 TP, TN, FP, FN

    for caption, pred_fake_words, true_fake_pos in zip(captions, pre_words, fake_text_pos_list):
        # 1. Tokenize 句子
        tokenized_caption = tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        tokens = tokenizer.tokenize(caption)  # 获取分词后的 token
        attn_mask = tokenized_caption.attention_mask[:, 1:].squeeze(0)  # 忽略 CLS token
        token_label = attn_mask.clone()

        # 2. 初始化 token 级别标签（真实标签）
        token_label[token_label == 0] = -100  # Padding 部分设为 -100
        token_label[token_label == 1] = 0      # 先全部设为 0

        # 3. 处理 fake_text_pos_list
        word_tokens = caption.split()  # 按空格拆分得到单词列表
        word_to_token_map = {}  # 记录 word->token 的映射（适用于 tokenizer 可能会拆分单词）
        token_index = 0

        for word_idx, word in enumerate(word_tokens):
            word_tokenized = tokenizer.tokenize(word)  # 该单词的 token 切分
            word_to_token_map[word_idx] = list(range(token_index, token_index + len(word_tokenized)))
            token_index += len(word_tokenized)  # 更新 token 位置索引
        
        # 4. 根据 fake_text_pos_list 设置 token_label
        for word_idx, is_fake in enumerate(true_fake_pos):
            if is_fake == 1 and word_idx in word_to_token_map:
                for tok_idx in word_to_token_map[word_idx]:
                    token_label[tok_idx] = 1  # 该单词对应的 token 都标记为假单词
        
        # 5. 处理 pre_words（预测的假单词）
        pred_fake_words = set(pred_fake_words.split(", ")) if pred_fake_words else set()
        pred_token_label = torch.zeros_like(token_label)

        for word in pred_fake_words:
            for word_idx, original_word in enumerate(word_tokens):
                if original_word == word and word_idx in word_to_token_map:
                    for tok_idx in word_to_token_map[word_idx]:
                        pred_token_label[tok_idx] = 1  # 预测的 token 设为 1（假单词）

        # 6. 计算 TP, TN, FP, FN
        valid_mask = token_label != -100  # 只计算有效 token
        TP += torch.sum((token_label == 1) & (pred_token_label == 1) & valid_mask).item()
        TN += torch.sum((token_label == 0) & (pred_token_label == 0) & valid_mask).item()
        FP += torch.sum((token_label == 0) & (pred_token_label == 1) & valid_mask).item()
        FN += torch.sum((token_label == 1) & (pred_token_label == 0) & valid_mask).item()

    # 7. 计算 Accuracy
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    return accuracy



def evaluate_model(test_loader, model, processor,device,option_vectors,vectorizer,options,option_labels,tokenizer):

    IOU_pred = []
    token_acc_list = []
    cls_nums_all = 0
    cls_acc_all = 0  
    val_item_count = 0 
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for inputs, batch_answers,fake_text_pos_list,captions,fake_image_box in tqdm(test_loader, desc="Evaluating"):
        
    
        # generated_texts = run_batch(inputs)
        val_item_count += len(batch_answers)
        ## model是DistributedDataParallel包装后的类，并没有generate方法，如需调用，应该使用model.module调用基础模型后再调用generate()方法
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        ###解析得到模型的文本输出
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
        task_answers = []
        pred_words_list = [] # 存储预测的假单词
        output_coords = torch.zeros((len(generated_texts), 4)).to(device)
        true_coords = torch.zeros((len(generated_texts), 4)).to(device)
        
        
        for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):

            full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
            pre_words = [] # 初始化pre_words
            ## 如果在返回中有Swapped words:，先拿出来备用
            if 'Swapped words:' in full_answer:
                pre_words = full_answer.split('Swapped words:')[-1]
                full_answer = full_answer.split('Swapped words:')[0]
            
            if '<loc_' in full_answer:
                task_answers.append(full_answer.split('Manipulated face')[0])
                output_coords[i] = parse_coordinates(full_answer).to(device)
                true_coords[i] = parse_coordinates(answers).to(device)
            # 将 output_coord 堆叠到 output_coords中
            else:
                task_answers.append(full_answer)
                true_coords[i] = parse_coordinates(answers).to(device)
            
            pred_words_list.append(pre_words)
                
        
                
        real_multi_label, real_label_pos = get_multi_label(batch_answers,device)
        real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
        real_label[real_label_pos] = 0
        best_options, _ ,best_multi_labels,pred_label =  get_best_option(task_answers, option_vectors,vectorizer,options,option_labels,device)
        
        ##--reeal/fake---##
        cls_nums_all = val_item_count
        cls_acc_all += torch.sum(real_label == pred_label).item()
        
        IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)

        # IOU_pred.extend(IOU.cpu().tolist())
        for iou_value in IOU.cpu().tolist():
            if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                IOU_pred.append(iou_value)
            else:
                IOU_pred.append(0.0)
        ######################################3

        ##-multi--##
        multi_label_meter.add(best_multi_labels, real_multi_label)
        ## token-acc ##
        token_acc_list.append(compute_token_acc(captions,pred_words_list,fake_text_pos_list,tokenizer))
        

    IOU_score = sum(IOU_pred)/len(IOU_pred)
    Token_ACC = sum(token_acc_list)/len(token_acc_list)
    ACC_cls = cls_acc_all / cls_nums_all
    
    MAP = multi_label_meter.value()[:3].mean()
    

    return ACC_cls, cls_acc_all, cls_nums_all,IOU_score,MAP,Token_ACC


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on multiple datasets")
    parser.add_argument("--GPU_nu", type=int, default=0, help="GPU number to use")
    parser.add_argument("--model_id", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for evaluation")
    parser.add_argument("--vals", nargs='+', required=True, help="List of validation JSON file paths")
    parser.add_argument("--output_file", type=str, required=True, help="File to save evaluation logs")
    parser.add_argument("--tokenizer", type=str, default='bert-base-uncased', help="Tokenizer pth")
    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.GPU_nu}" if torch.cuda.is_available() else "cpu")

    # Load model & processors
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).eval().cuda().to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Fixed options & vectorizer
    options = [
        "A. No.",
        "B. Only face swap.",
        "C. Only face attribute.",
        "D. Only text swap.",
        "E. Face swap and text swap.",
        "F. Face attribute and text swap.",
    ]
    
    option_labels = [
    torch.tensor([0, 0, 0, 0]).to(device),
    torch.tensor([1, -0.33, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, 1, -0.33, -0.33]).to(device),
    torch.tensor([-0.33, -0.33, 1, -0.33]).to(device),
    torch.tensor([0.5, -0.5, 0.5, -0.5]).to(device),
    torch.tensor([-0.5, 0.5, 0.5, -0.5]).to(device),
    ]
    vectorizer = TfidfVectorizer().fit(options)
    option_vectors = vectorizer.transform(options).toarray()

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Logging to file
    def log_print(*args_, **kwargs_):
        print(*args_, **kwargs_)
        with open(args.output_file, "a") as flog:
            print(*args_, **kwargs_, file=flog)
    
    def collate_fn(batch):

        #### DGM4的定义：
        images, questions, answers, fake_words_lists, captions,fake_image_box = zip(*batch)
        
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        return inputs, answers,fake_words_lists,captions,fake_image_box

    log_print(f"test model_id is {args.model_id}")

    for val_js in args.vals:
        with open(val_js, "r") as f:
            val_data = json.load(f)

        test_dataset = OriDGM4Dataset(split="validation", data=val_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            prefetch_factor=None,
        )

        ACC_cls, cls_acc_all, cls_nums_all, MAP, IOU_score, Token_ACC = evaluate_model(test_loader,model,processor,device,option_vectors,vectorizer,options,option_labels,tokenizer)

        log_print('#######<--record-->###########')
        log_print(f"ACC_cls (Accuracy): {ACC_cls*100} (cls_acc_all: {cls_acc_all}, cls_nums_all: {cls_nums_all})")
        log_print(f"MAP (Mean Average Precision): {MAP*100}")
        log_print(f"IoUscore: {IOU_score*100}")
        log_print(f"Token_Acc: {Token_ACC*100}")
        log_print('########<--record-->#########')
        log_print("END############################################################################################")

    print(f"log at {args.output_file}")


if __name__ == "__main__":
    main()
