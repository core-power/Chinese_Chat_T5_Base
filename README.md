中文版对话机器人

在1300w+问答和对话数据上做有监督预训练

## 训练硬件和时间
4*Titan RTX,耗时10天

## 更新进度
model v1 :2023.3.12

model v2 :2023.3.12（知识增强）

## 注意事项

1、请使用下面方式调用模型输出结果，Hosted inference API的结果因为我无法修改后台推理程序，不能保证模型输出效果，只是举了两个例子展示。

2、模型采用top p的解码方式，每次运行可能结果都略微有些不同。

3、目前模型还是初步迭代完2epoch，数据种类和数据量现阶段算是比较少模型效果的话勉勉强强，后续还会加入更多数据进行迭代优化，到时候会更新。

4、因为数据丰富度限制和模型容量限制可能在某些问题上回答不相关现象，模型现在有着跟chatgpt一样的缺点就是会在关键信息错误，有点乱编出信息来。

5、模型对于有些百科知识会答不准的问题，是因为缺少相关百科知识增强，目前也正在爬取百度百科进行知识增强。


## 模型推理

Install package:
```
pip install transformers 
```

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("mxmax/Chinese_Chat_T5_Base")
model = AutoModelForSeq2SeqLM.from_pretrained("mxmax/Chinese_Chat_T5_Base") 
device = 'cuda' if cuda.is_available() else 'cpu'
model_trained.to(device)
def postprocess(text):
  return text.replace(".", "").replace('</>','')

def answer_fn(text, top_p=0.6):
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device) 
  out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=512,temperature=0.5,do_sample=True,repetition_penalty=6.0 ,top_p=top_p)
  result = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(result[0]) 
text="宫颈癌的早期会有哪些危险信号"
result=answer_fn(text, top_p=0.6)
print('prompt:',text)
print("result:",result)
```

## 预测示例
#### 普通百科问答
```bash
Input:
关于基金啥叫前端申购代码,后端申购代码.有啥区别.有甚么作用.
Model output:
基金公司一般会根据投资者的申购代码进行操作。 前端：指在购买时，通过网上银行或手机客户端等渠道输入“赎回”信息后(如需提前确认),将资金转入到证券账户中;而后的端:是指对已持有该股票并已经卖出的资金做委托
```

```bash
Input:
AI是什么？
Model output:
AI是人工智能的简称，它是一种基于计算机视觉和信息处理技术(如图像识别)而进行的一种智能计算。在人类历史上有无数个发明者诞生于现代科技中： 比如我们现在所熟知的一个“机器人”就是用一个机器来模拟人脑中的各种生物行为模式并加以控制与改造的过程
```

```bash
Input:
chatgpt的出现会导致程序员失业吗？
Model output:
会的，因为chatgpt是程序员必备的工具。
```

```bash
Input:
什么是无公害蔬菜
Model output:
无公害蔬菜 一、什么是“纯天然”？ “天然无污染的植物，是大自然中最珍贵的食物。它不仅具有观赏价值和药用价值的双重功效;而且还含有多种维生素及矿物质等营养成分：如绿叶菜中的胡萝卜素(C)与红萝卜中所含丰富的铁质以及各种微量元素硒元素等等!这些营养素在自然界里都可被人体吸收利用并转化成有机酸或无机盐来食用因此,人们经常把它们称为"绿色食品"、"绿色食品”、“健康水果\保健产品
```

```bash
Input:
什么是染色体遗传学说
Model output:
染色体遗传是遗传学的一个分支，它与基因学、生物学和分子生物学等有关。 在人类基因组中,有6个细胞分裂成1种或2型其中3系的分化为4条线(即：5)/8=7;而9属中的10只则分为12对: 1组共生一个DNA序列 2一组同源性RNA 3一对单克隆体 4一共有16支相同类型的抗原 5两组成不同的核苷酸 6二位互补性的抗体 7三者相互结合形成新的免疫复合物
```

#### 医学领域问答
```bash
Input:
重感冒的话吃什么药
Model output:
你好，根据你的描述属于病毒感染引起的症状,需要增加营养补充维生素微量元素，易消化饮食。可以服用感冒药如复方氨酚脘胺颗粒治疗看看
```

```bash
Input:
胃癌如何筛查
Model output:
你好,胃癌的诊断标准是：1早期确诊，明确病因。2根据病史、病理类型及有无并发症等综合分析;3结合临床症状和体征进行初步判断是否为癌前病变或有肿瘤变现趋势者(如无明显变化) 4对患者的情况做进一步检查以确定治疗方案
```

```bash
Input:
甲状腺五类结节怎么办
Model output:
甲状腺结节是一种常见的良性肿瘤，一般无症状,但有恶变的可能建议您到医院做一下B超检查和胸部X线片等进一步明确诊断后对症治疗。
```

```bash
Input:
宫颈癌的早期会有哪些危险信号
Model output:
你好，宫颈癌早期症状：1、白带增多、粘稠;阴道分泌物多或呈脓性。2、有异味及血丝;2、月经量少;3、腰酸背痛;4、乳房胀满;5、乳头溢液;6、下腹坠涨;7、肛门疼痛等
```

#### 生活问答
```bash
Input:无线路由不能上网
Model output:
 1、检查信号是否正常;2、号卡是否有欠费;3、如上面2项都已超出可使用，则说明是号码存在问题。建议您联系归属地客服查询一下;4、也可以到营业厅进行缴费试试看;如果在异地无法正常使用宽带上网功能的话，可以拨打1#11转人工台报障处理的
```

```bash
Input:为什么电脑老是自动重启
Model output:
建议你下载恶意软件和木马强杀工具windows清理助手查被杀毒，修复注册表。然后在开机按F8进入安全模式后选择重启或关机再试! 1、请你用系统自带的系统的还原：设置->程序与附件中找到“恢复出厂设定”/勾选为自动完成(如果故障依旧);2、双击打开命令提示符输入SFC+ALT回车;3、点击"开始扫描,并执行此操作;4、磁盘整理;5、检查是否使用其他盘片代替了操作系统;如以上方法仍无法解决你的问题请将机器送到三*服
```

```bash
Input:
深圳哪家店火锅好吃？
Model output:
火锅店 地址：深圳宝安区沙井路20号(深大对面)。
```

```bash
Input:
北京有哪些景点方好玩呢推荐下
Model output:
在北四环附近，有一家叫“北京动物园”的野生动物园。 地址：北京市朝阳区东城区西三环南口(原昌平区)
```

```bash
Input:
儿子过生日，我该送些什么礼物呢？
Model output:
送个礼物，就是给宝宝买一个玩具。 比如：小红帽、小熊抱枕(可以自己DIY)等
```

#### 瞎编故事系列
```bash
Input:
齐桓公如何解释呢
Model output:
齐桓公是春秋时期楚国的一位君主，他的父亲名叫韩信。公元前479年(前227),楚王赵匡胤即位后不久就与秦国相会了;公元398年被封为“燕侯”、晋国的开国元勋——郑玄之孙梁惠帝和周武灵君等五人联合讨伐项羽时被刘邦所杀!
```


## 引用
```bash
@misc{mxmax,
  title={chinese_chat: Chinese_Chat_T5_Base},
  author={Ma Xin},
  year={2023},
  howpublished={\url{https://github.com/core-power/Chinese_Chat_T5_Base}},
}
```
