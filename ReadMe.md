# DeepSeek的LLM学习

这是一篇学习笔记，学习内容是现阶段LLM的一些架构和工具梳理；起因是DeepSeek的爆火让我想瞅瞅它的源码，但是从Github一下载，发现围绕LLM涌现的新工具方法好多；还好有之前博士期间做的NLP相关工作，有一定的基础，希望能通过这篇文章整理一下新工具方法，也能唤醒我过去的记忆，有不足的地方请各位大佬指正^_^

以下先介绍我对新工具的学习认知，再记录一个蒸馏DeepSeek网络的小项目。

## 开发工具学习

### HuggingFace

DeepSeek在GitHub上模型链接既是HuggingFace，先上一段官方介绍：

> The platform where the machine learning community collaborates on models, datasets, and applications. 		

个人理解既是Github在Machine Learning领域的特供版。这可太解决ML研究的痛点了，进行模型训练，验证的时候难免会用到他人的模型网络以及数据库，之前读博时各种论文发表的数据库格式千奇百怪，前处理非常消耗时间，当时就想着有没有统一的格式去除这种无意义的dataset preprocess . 

HuggingFace 同步发布了很多预处理的库，例如Transformers，Datasets等，都可使用pip安装

> pip install transformers datasets

这些库提供了快速数据预处理，模型微调，模型训练的API，这真的太赞了！

个人理解现在LLM大致的开发框架如下，请大佬指正：

![](https://image-kdp777.obs.cn-north-4.myhuaweicloud.com/img/LLM_structure.png)

PS. 国内有一个类似于HuggingFace的魔塔社区，DeepSeek源码可以通过魔塔社区不使用科学上网的方式下载；

### Jupyter

Jupyter 是一个python的IDE工具，可实时运行，开启也很快，并且可以实现远程登陆使用，是神器！！！

![](https://image-kdp777.obs.cn-north-4.myhuaweicloud.com/img/Jupyter_IDE.png)

.ipynb是Jupyter的文件后缀，Jupyter NoteBook基于本地运行，Jupyter Lab可以联网编辑。

###  LLaMA-Factory

官方解释：LLaMA Factory 是一个简单易用且高效的大型语言模型（Large Language Model）训练与微调平台。通过 LLaMA Factory，可以在无需编写任何代码的前提下，在本地完成上百种预训练模型的微调，框架特性包括：

- 模型种类：LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi 等等。
- 训练算法：（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO 训练、DPO 训练、KTO 训练、ORPO 训练等等。
- 运算精度：16 比特全参数微调、冻结微调、LoRA 微调和基于 AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ 的 2/3/4/5/6/8 比特 QLoRA 微调。
- 优化算法：GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ 和 PiSSA。
- 加速算子：FlashAttention-2 和 Unsloth。
- 推理引擎：Transformers 和 vLLM。
- 实验监控：LlamaBoard、TensorBoard、Wandb、MLflow、SwanLab 等等。

安装方式：

>git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
>
>cd LLaMA-Factory
>
>pip install -e ".[torch,metrics]"

个人理解是希望通过配置脚本或者GUI的模式，将成熟的微调方式和通用模型打包起来；在训练，微调和蒸馏领域有较多的使用，后续将结合具体的例子展示LLaMA-Factory的初级使用方法。



## DeepSeek 蒸馏小项目

本次的蒸馏小项目依据**九天Hector**老师的[DeepSeek蒸馏课程](https://www.bilibili.com/video/BV1X1FoeBEgW?buvid=XUE8644E325CF60308A32B1128162ACA89E7F&from_spmid=main.space-search.0.0&is_story_h5=false&mid=arl4wKR4RBwjc1Xn3l2j8w%3D%3D&plat_id=114&share_from=ugc&share_medium=android&share_plat=android&share_session_id=4742a209-f15f-4a13-94fd-faa0b4258a72&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1741683894&unique_k=pPssA00&up_id=385842994&share_source=weixin) ;

本人借助该项目进行现有LLM开发工具技术的学习。如课程中所说，DeepSeek蒸馏大致分为以下几个步骤：

1. 创建虚拟环境，具体依托于conda工具，在conda创建的虚拟环境中进行模型的蒸馏开发；

2. 下载蒸馏知识的学生模型- Qwen2.5-1.5B-Instruct，下载链接为[QWen-Student-Model](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/)

   * 学生模型文件中的safetensors文件，存储的是模型的参数数据，是HuggingFace推出的高效模型参数存储模式，pytorch可读取其为model.state_dict的数据类型，配合构建的model类使用，测试代码请参见[GitHub链接](https://github.com/KDP777/LLM_Learning/blob/main/CNN_safetensorsTest.ipynb).
     * transformers库中集成了大量已经成熟的网络模型，具体可到https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/中的MODELS中查询，Qwen2已经集成到了transformers库中，可使用transformers.Qwen2ForCausalLM直接获取网络结构，并载入相应的safetensors的参数数据，一小段测试代码可供参考[ref code](https://github.com/KDP777/LLM_Learning/blob/main/LoadQwenModel.ipynb);
     * 自定义的网络结构，需要自己编写网络结构的py文件注册网络，可见DeepSeek定义的网络结构文件[modelling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py)
   * config.json 定义了模型的基本参数，重点是model_type,告知了Transformers库要载入的模型类；AutoModelForCausalLM函数需要该文件的存在;
   * generation_config.json 是模型推理时的建议的模型参数设置，包括temperature和top_p,表示模型生成结果的随机性;
   * tokenizer.json和tokenizer_config.json定义了分词器的模型和参数，负责将文本转化为数字矩阵，类似于早期的Word2Vec等，在调用Transformers的AutoTokenizer.from_pretrained会用到;

3. 整理清洗数据集

   * 收集公开的数据集，包括:

     * [Numina COT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT):  包含大约86万个数学问题和解决方案的集合，所有解决方案都采用了“Chain of Thought”（CoT）推理方式;
     * [APPS Dataset](https://huggingface.co/datasets/codeparrot/apps): Automated Programming Problem Solving，是一个用于编程任务的基准数据集，包含了10,000
       个问题，旨在评估语言模型生成代码的能力，特别是从自然语言描述中生成代码的能力。它为基于自然语言的代码生成、自动编程和相关任务的研究提供了一个标准化的测试平台。
     * [TACO（Text to Code）](https://huggingface.co/datasets/BAAI/TACO): 用于评估语言模型从自然语言描述生成代码能力的基准数据集。它包含了26,443个编程问题，配有Python代码解决方案，适用于代码生成、自然语言理解和问题求解等任务。
     * [long_form_thought_data_5k](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k): 它存储了 JSONL (JSON Lines) 格式的数据，每一行都是一个包含多个键值对的字典; 问题可以是来自不同领域的各种类型，如数学、物理、化学等。该字段包含了模型的长篇思维过程和最终解决方案，分为两个部分: thought：模型的长时间思考过程。这部分通常包含模型对问题的分析、推理过程，可能会涉及到中间步骤、假设、推理的推导等; solutions: 模型给出的最终解决方案，这部分是针对问题的最终答案。

   * 让 DeepSeek R1 对公开数据集进行全部的回答，即输入数据集问题，输出思维链（CoT）和模型推理结果；

     ![](https://image-kdp777.obs.cn-north-4.myhuaweicloud.com/img/Dataset_clean.png)

   * 输出结果格式化：获得训练数据列表文件后，将他们转换为统一的格式（这一步使用DeepSeek V3进行重写，输出较长，处理数据大约要花费20-30元，*这一部分不太理解，个人猜想：将这些输出的答案转换成易于训练的数据格式，比如将这一大段CoT转成ii步骤1,步骤2...*）

   * 人工对模型输出的结果进行再筛选，选择正确的放到后续进行全量指令微调；

     ***数据整理和筛选是蒸馏的核心，本文重点于学习蒸馏的全步骤，不过多赘述，使用的是九天Hector整理好的数据集，链接如下***[](https://data-hub.obs.cn-north-4.myhuaweicloud.com/Distill_data_17k-train.arrow)

   4. 在LLaMa Factory中注册自定义数据集

      > 在LLaMa Factory文件夹中的data/data_info.json文件末尾，加上以下段落：
      >
      > "Distill":{
      > "file_name":"/home/kdp/WorkStation/DeepSeek/Studio/Distill/Distill_data_17k-train.arrow",
      > "formatting":"sharegpt",
      > "columns": { "messages": "conversations", "system": "system" },
      > "tags": { "role_tag": "from", 
      >     "content_tag": "value", 
      >     "user_tag": "user",
      >     "assistant_tag": "assistant" 
      >   }
      > }

      Arrow数据集是HuggingFace提供的二进制数据集格式，可用HuggingFace提供的Datasets工具直接打开进行解码;

      ```python
      from datasets import load_datasets
      dataset = load_dataset('path_to_file.arrow');
      print(dataset["train"][0]); #显示dataset trainData的第一个用例
      ```

   5. 编写全量指令微调的LLaMA Factory脚本;

      * 全量微调（Full Fine-tunning）：在特定数据数据集上进行训练，模型的全部参数都进行更新；

      * 高效参数微调（Parameter-Efficient Fine-tunning）：只更新一部分参数或通过对参数进行结构化约束，例如稀疏化，低秩近似来降低微调的参数量；

      * 指令微调（Instruction Tunning）：特殊点在于数据集的结构，即每一个输入输出对配有一个人类指令，可视为有监督微调（SFT）的一种特殊形式，专注于理解和遵循人类指令来增强LLM的能力与可控性（*个人理解是数据集中的system关键字，告知模型回答该问题的方向*）

        > 指令微调数据集大致结构：
        >
        > {
        >
        > "instruction/system": Question,
        >
        > "input": 中国的首都是哪个城市？
        >
        > "output":中国的首都是北京。
        >
        > }

      * LLaMA Factory 脚本如下：

        > ###model
        > model_name_or_path: /home/kdp/WorkStation/DeepSeek/Studio/Qwen2.5-1.5B-Instruct
        >
        > ###method
        > stage: sft
        > do_train: true
        > finetuning_type: full
        > deepspeed: examples/deepspeed/ds_z3_offload_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
        >
        > ###dataset-提前注册好的数据集
        > dataset: Distill
        > template: qwen
        > cutoff_len: 8192
        > max_samples: 100000
        > overwrite_cache: true
        > preprocessing_num_workers: 16
        >
        > ###output
        > output_dir: ./saves/MiniDeepSeekR1/full/original
        > logging_steps: 1
        > save_steps: 100
        > plot_loss: true
        >
        > ###train
        > per_device_train_batch_size: 1
        > gradient_accumulation_steps: 12
        > learning_rate: 1.0e-5
        > num_train_epochs: 3.0
        > lr_scheduler_type: cosine
        > warmup_ratio: 0.1
        > bf16: true
        > ddp_timeout: 180000000

    * LLaMa Factory 运行微调脚本

     > cd path_to_LLaMA Factory
     >
     > FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train examples/train_full/qwen2_full_sft.yaml

​	本机性能过弱，无法运行大模型的training 过程，只能把教程中的结果贴上来供大家参考:

​	![](https://image-kdp777.obs.cn-north-4.myhuaweicloud.com/img/Distill_process.png)

7. 运行微调后的模型，大致流程如下，主要依托transformers加载库，并进行推理：

   * 加载model和tokenizer，使用的是DeepSeek官方蒸馏好的Qwen模型，下载链接:[](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)；
   * 使用pipeline将tokenizer和model串起来；
   * 准备好prompt，输入pipeline，生成结果；
   * 解析结果，获得回答；

   具体请见模型推导程序[Github链接](https://github.com/KDP777/LLM_Learning/blob/main/Model_inf.py).
