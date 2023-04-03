# 加载训练后的模型
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("./outputs/model_files")
model_trained = AutoModelForSeq2SeqLM.from_pretrained("./outputs/model_files") #./v1/model_files
# import torch
# from transformers import AutoTokenizer
# 修改colab笔记本设置为gpu，推理更快
device = 'cuda' if cuda.is_available() else 'cpu'
model_trained.to(device)
def preprocess(text):
  return text.replace("\n", "_")
def postprocess(text):
  return text.replace(".", "").replace('</>','')

def answer_fn(text, sample=False, top_k=50):
  '''sample：是否抽样。生成任务，可以设置为True;
     top_p：0-1之间，生成的内容越多样、
  '''
  # text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device) 
  if not sample: # 不进行采样
    out = model_trained.generate(**encoding, return_dict_in_generate=True, max_length=512, num_beams=4,temperature=0.5,repetition_penalty=10.0,remove_invalid_values=True)
  else: # 采样（生成）
    out = model_trained.generate(**encoding, return_dict_in_generate=True, max_length=512,temperature=0.6,do_sample=True,repetition_penalty=3.0 ,top_k=top_k)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  if out_text[0]=='':
    return '我只是个语言模型，这个问题我回答不了。'
  return postprocess(out_text[0]) 
# text="人工智能"
text_list=[]
while True:
  text = input('请输入问题:')
  result=answer_fn(text, sample=True, top_k=100)
  print("模型生成:",result)
  print('*'*100)



      