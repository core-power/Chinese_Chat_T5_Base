# -*- coding: utf-8 -*-
# 引入相应的包 Importing libraries
import os,json
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration,BertTokenizer
from torch.utils.data import WeightedRandomSampler

from rich.table import Column, Table
from rich import box
from rich.console import Console
from tqdm import tqdm
import time
import jsonlines
from pandas.io.json import json_normalize
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from torch.nn import DataParallel
# Setting up the device for GPU usage
from torch import cuda
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
device = 'cuda' if cuda.is_available() else 'cpu'

smooth = SmoothingFunction().method1
# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    # console.print(table) # TODO TODO TODO 

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    """
    多卡负载均衡
    """
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        # print('len(inputs): ', str(len(inputs)))
        # print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        # print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        if len(inputs) > 0:
            bsz = inputs[0].size(self.dim)
        elif kwargs:
            bsz = list(kwargs.values())[0].size(self.dim)
        else:
            raise ValueError("You must pass inputs to the model!")
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        # print('bsz: ', bsz)
        # print('num_dev: ', num_dev)
        # print('gpu0_bsz: ', gpu0_bsz)
        # print('bsz_unit: ', bsz_unit)
        # print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)

class DataSetClass(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(device, dtype = torch.long),
            "source_mask": source_mask.to(device, dtype = torch.long),
            "target_ids": target_ids.to(device, dtype = torch.long),
            "target_ids_y": target_ids.to(device, dtype = torch.long),
        }
    
def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    训练模型

    """
    model = BalancedDataParallel(14 // 2, model, dim=0).cuda()
    # model = DataParallel(model)
    model.cuda()
    model.train()
    time1=time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous() 
        lm_labels = y[:, 1:].clone().detach() 
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 
        ids = data["source_ids"].to(device, dtype=torch.long) 
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0].mean()
        # 每100步打印日志
        if _ % 100 == 0 and _!=0:
            time2=time.time()
            print(_,"epoch:"+str(epoch)+"-loss:"+str(float(loss))+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(tokenizer, model, device, loader):
    """用BLEU4评估"""
    model.eval()
    bleus = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0), desc='Evaluate'):
            target_ids = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=model_params["MAX_SOURCE_TEXT_LENGTH"], 
                do_sample=True, 
                top_p=0.6,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in target_ids]
            bleu_score = sentence_bleu([list(p) for p in preds],list(target[0]),[0.25, 0.25, 0.25, 0.25])            
            bleus.append(bleu_score)
            return sum(bleus) / len(bleus)
        
# t5模型训练
def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):
    """
    T5 trainer
    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])#rT5Tokenizer

    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    console.log(f"[Data]: Reading data...\n")

    dataframe = dataframe[[source_text, target_text]]#, 'type'

    train_size = 0.94
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    
    # 打印数据集相关日志：数据量、训练步数
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")
    total_train_steps=int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"])/model_params["TRAIN_BATCH_SIZE"])
    console.print(f"Total Train Steps: {total_train_steps}\n")

    training_set = DataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = DataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    train_params = {
        # "sampler":train_sampler,
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    console.log(f"[Initiating Fine Tuning]...\n")
    best_bleu = 0
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        
        path = os.path.join(output_dir, "model_files")
        tokenizer.save_pretrained(path)
        console.log(f"[Initiating Validation]...\n")
        with torch.no_grad(): 
            cur_bleu = evaluate(tokenizer, model, device, val_loader)
            if cur_bleu > best_bleu:
                console.log(f"[Saving Model]...\n")
                model.save_pretrained(path)
                best_bleu = cur_bleu
            print('Best bleu: {}, Current bleu: {}'.format(best_bleu, cur_bleu))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

if "__main__" == __name__:
    # 定义模型的参数 
    model_params = {
        "MODEL": "./tmp_model/model_files",  # 初始模型路径
        "TRAIN_BATCH_SIZE": 62,  # 训练batch size
        "VALID_BATCH_SIZE": 62,  # 评估batch size
        "TRAIN_EPOCHS": 5,  # 训练epoch数
        "LEARNING_RATE": 1e-4,  # 学习率
        "MAX_SOURCE_TEXT_LENGTH": 256,  # 句子最大长度
        "MAX_TARGET_TEXT_LENGTH": 256,  # 标签最大长度
        "SEED": 42,  # 随机种子
    }

    input=[]
    target=[]
    with jsonlines.open('./data/belle_open_source_1M.train.json','r') as f:
        for l in f:
                input.append(l['input'])
                target.append(l['target'])
    df = pd.DataFrame()
    df['input']=input
    df['target']=target
    print("df.shape:",df.shape)
    T5Trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="./outputs",
    )