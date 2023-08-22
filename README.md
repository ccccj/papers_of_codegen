# papers_of_codegen [![My github](https://skillicons.dev/icons?i=github)](https://github.com/ccccj/papers_of_codegen) 


Some paper about code generation.

> If you have any papers you'd like to recommend, you can contact me at :envelope: 839808679@qq.com



# 一、大模型汇总（LLMs Summary）
## 1.1 开源模型
- ### [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf)

  :date: Sat, 13 May 2023

  :school: Author Affiliation : Salesforce AI Research

  :open_file_folder: Model Size: 220M, 770M, 2B, 6B, 16B
  
- ### [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf)

  :date: Thu, 2 Sep 2021

  :school: Author Affiliation : Salesforce Research Asia, Nanyang Technological University(Singapore)

  :open_file_folder: Model Size: CodeT5-small (60M) , CodeT5-base (220M)

  许多现有的方法把 code 看作是像自然语言这样的 token 序列，只是在其上采用传统的 NLP 预训练技术，这在很大程度上忽略了代码中丰富的结构性信息，而这对于完全理解代码的语义至关重要。

  CodeT5 是基于 T5 改进的。T5 的预训练任务是一个"去噪自编码"（denoising autoencoding）任务，即从输入文本中随机遮掩一部分 tokens（称为噪声），然后让模型基于遮掩后的输入文本预测原始文本。T5 模型的核心思想是将各种自然语言处理任务统一为一个 "文本到文本"（text-to-text）的框架。这意味着 T5 将所有任务视为输入文本的转换，通过输出文本来完成任务。

  CodeT5 的输入是双模态的，即注释和代码拼接起来再编码的。在预训练阶段，有以下任务：
  1. MSP（Masked Span Predicion）: 对于输入序列，sample 15% 的 token 给 mask 掉，在 mask 的过程中确保是掩盖整个单词。输出是预测被 mask 掉的 token;
  2. IT（Identifier Taggng）: 因为标识符在代码中有着很高的地位，为了让模型能够识别哪个是标识符，因此加入了标识符打标任务；
  3. MIP（Masked Identifier Prediction）: 基于“对一个变量，改变它的名字并不会改变它在代码中作用”这一个思想，设计了这个任务。首先把输入的代码中所有标识符都 mask 掉，但是同一个名字的标识符要用同一个 mask，在 Output 中去预测被 mask 前是什么标识符。；
  4. Bimodal Dual Generation : 在之前几项任务中，解码器只能够看到离散的 mask，而实际的下游任务是需要生成一个流畅的代码或者自然语言描述。为了减小这种差异，在输入序列中 mask 掉整个代码或者自然语言，并且让解码器复原。
  
## 1.2 闭源模型
- ### [PANGU-CODER2: BOOSTING LARGE LANGUAGE MODELS FOR CODE WITH RANKING FEEDBACK](https://arxiv.org/pdf/2307.14936.pdf)

  :date: Thu, 27 Jul 2023

  :school: Author Affiliation : Huawei Cloud, Chinese Academy of Science, Peking University

  :open_file_folder: Model Size: 15B
  
- ### [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644.pdf)

  :date: Tue, 20 Jun 2023
  
  :school: Author Affiliation : Microsoft Research

  :open_file_folder: Model Size: 1.3B


# 二、Code 论文（Recent Innovative Papers）
## 2.1 模型水印
- ### [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226v3.pdf)  【ICML 2023】

  :date: Tue, 24 Jan 2023

  :school: Author Affiliation : University of Maryland
  
  大型语言模型(LLMs)的发展促进了人类对工具的使用，但是也会导致更多的恶意攻击，如利用AI操控选举活动，制造假新闻，制造假数据，以及在学术写作和编码作业中作弊。让LLMs生成的文本有水印，可以极大程度的避免这些问题。
  
  作者提出了一种有效的水印，可以在短跨度的token（少至 25 个token）中检测到合成文本，而误报（人类文本被标记为机器生成）在统计上是不可能的。水印检测算法可以被公开，使第三方（例如社交媒体平台）能够自行运行，也可以保持私有，并在 API 上运行。

  该水印有以下特性：
  1. 水印检测不需要通过调用  LLM API 或者获知 LLM 参数就可以实现模型
  2. 不需要额外训练就以对文本添加水印
  3. 即便只有很小的一段生成文本，水印也可以被检测
  4. 除非大幅修改生成文本，水印无法被移除
  5. 针对水印检测，可以计算出一个严格的统计量
  6. 水印对人类肉眼来说是不可见的

  作者提出的两种水印方法：hard watermarking 和 a more sophisticated watermark。

  hard watermarking 总体思路是，将上一个词的哈希值作为 seed，使用这个 seed，将词汇表随机划分为大小相等的“绿名单”G 和“红名单”R，即，词汇表中的任意一个词都会被随机到 G 或 R。下一个生成的词，在 sample 的时候只会去绿名单中sample。但对于某些固定搭配，基本上只会以 AB 的形式出现，一旦 B 存在于红名单中，A 后面只能搭配绿名单的词，无论选什么词都会使得单词搭配出现问题。作者提出的第二种方法相当于“建议”它选择率列表中的词，而不是强硬的规定。
  
- ### [Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/pdf/2305.15060.pdf)

  :date: Wed, 24 May 2023

  :school: Author Affiliation : Seoul National University 2NAVER AI Lab 3NAVER Cloud

  作者发现，以往对于文本的水印生成方法（如分成两组token，优先选择绿色token），对于代码生成任务来说不管用，因为模型输出的质量会受到严重影响。
  
  在这项工作中，作者提出了一种新的水印方法 —— SWEET（Selective WatErmarking via Entropy Thresholding），作者不对代码输出中的每个token应用绿-红规则，而是只选择性地将水印应用于熵值足够高、超过规定阈值的 token。这种选择性水印方法不会对 gold tokens 应用绿-红规则，以防止它们落入红色组，因此不会出现上述问题。

## 2.2 code debug
- ### [Language Models Can Teach Themselves to Program Better](https://arxiv.org/pdf/2207.14502v4.pdf)
  
  :date: Fri, 29 Jul 2022

  :school: Author Affiliation : Microsoft Research, MIT

  作者尝试让 LM 自己生成数据（编程问题和答案），然后由 Python 解释器对编程问题和答案进行验证，以确保正确性，最后再用自己生成的数据来微调自己。（即，自己生成问题和答案，通过验证的数据，再用来微调自己）

  步骤：
  1. 生成问题：问题是从训练集中随机抽取一组问题并将其拼接在一起，只有问题没有解决方案。然后提示 LM 生成与小型训练集类似的问题，由此来生成额外的问题。在检查生成的问题的语法有效性后，过滤删除一部分问题。
  2. 生成答案：使用 few-shot learning 让 LM 来生成解决方案。
  3. 验证解决方案。使用 python 解释器验证生成的解决方案。这一步骤不仅要求答案是正确的，还要求解决方案能在一定的时间内完成（时间复杂度不能太高）。
  4. 微调。LM 在这个过滤后的问题-解决方案对的合成数据集上进行了微调。
  5. 重复以上步骤。
  
  ![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/ca0ba0a6-ac8e-4658-ae73-70d1b4dac1da)

## 2.3 tokenization
- ### [CODEBPE: INVESTIGATING SUBTOKENIZATION OPTIONS FOR LARGE LANGUAGE MODEL PRETRAINING ON SOURCE CODE](https://arxiv.org/pdf/2308.00683.pdf)  【ICLR 2023】

  :date: Tue, 1 Aug 2023
  
  :school: Author Affiliation : Naver Labs Europe, University of Amsterdam

  考虑源代码的特定特征，确定用于 Code 任务的最有效、最省时的 subtokenization 方法。

  作者分别考虑了以下因素对代码生成、代码摘要、代码克隆检测、代码翻译任务的影响：
  1. 复合 token：自然文本中标点符号的一部分很小，因此它们在 subtokenization 中的分离不会对长度产生太大影响，而在源代码中，标点符号占字符的 12.8%，并且通常形成频繁的组合，将其加入复合 token 中可能会大大减少长度。此外，大量常用语句的存在是源代码的另一个特定特征，这些模式再次可能形成复合 token。作者研究使用复合 token 对性能和长度效率的影响；
  2. subtokenization 算法：不同的 subtokenization 算法会有不同的影响，论文中作者比对了两种算法—— BPE 和 UnigramLM；
  3. vocabulary size：不同的 vocabulary size 会对下游任务产生不同的影响；
  4. 编程语言之间的可移植性：研究使用在一种编程语言上训练的 subtokenizer 对另一种编程语言的影响；
  5. BPE-Dropout 方法的影响。


# 三、评测相关论文（Papers Related to Review）
## 3.1 评测框架
- ### [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation](https://arxiv.org/pdf/2305.01210.pdf)

  :date: Tue, 2 May 2023
  
  :school: Author Affiliation : University of Illinois Urbana-Champaign Nanjing University

  作者发现现有的 LLM-for-code benchmarks 存在以下共同的局限性:

  1. 测试不充分。只能为每个编码问题平均包含不到 10 个测试。此外，这些测试相对来说过于简单，无法充分探索代码的功能或角落案例。
  2. 问题描述不精确。代码生成的输入,除了函数签名外，还包括自然语言描述。现有 benchmarks 中的这些任务描述往往过于模糊，无法完全阐明预期的程序行为。例如，输入文档可能没有指定预期的输入域（例如，只有正整数）或函数应如何处理异常。因此，LLM 对此类编程问题的解释可能与实际测试不同，导致有能力的LLM被误判。
  
  作者建立了一个新的评估框架—— EvalPlus，用于改进现有的 code benchmarks，以更严格的标准来评估LLM生成代码的功能正确性。EvalPlus 扩展了 HUMANEVAL benchmarks，创建了 HUMANEVAL+，将测试用例规模提高了81倍。
  EvalPlus 把原始数据集的 ground-truth（真实代码）、作为示范的测试用例的输入和专门的指令来构建一个 prompt，以查询 ChatGPT 并生成一组高质量的种子输入。从这些种子输入开始，执行类型感知突变，快速生成大量新输入和种子输入。
  再使用差分测试来交叉检查 ground-truth（正确的代码）和 LLM 生成的解决方案的输出。除此之外，作者还设置了一个最小覆盖集，能基本全面覆盖案例，但数量不多，可以加速评估。


![EvalPlus](https://github.com/ccccj/papers_of_codegen/assets/33253635/fa7d8888-d1fa-4295-9007-4bc41de25850)

  作者使用 HUMANEVAL+ 对19种LLM以及各种温度设置进行了全面评估。使用 HUMANEVAL+ 与使用基本 HUMANEVAL 进行评估相比，所有 pass@k 成绩均持续下降。所有模型的 pass@k 平均降低了 13.6-15.3%。CodeGen 降低 18.5%、StarCoder 降低 14.1%、ChatGPT 降低 13.4％ 、GPT-4 减少13.8％.

- ### [A Simple, Yet Effective Approach to Finding Biases in Code Generation](https://arxiv.org/pdf/2211.00609.pdf)  【ACL 2023】

  :date: Mon, 31 Oct 2022
  
  :school: Author Affiliation : University of Warsaw, DeepMind, Google(University of Warsaw)

  该论文提出了一个框架，可用于测试代码生成模型的实际性能。该框架识别代码生成模型可能有用到的 cues，然后更改或删除这些线索以测试模型的推理能力。
  
  作者将代码示例视为三个区块（Blocks-Of-Influence）的组合，每个区块为模型提供不同的 cues。修改三种区块中的 1-2 种，如果会使原本的模型生成代码效果变差，说明模型并未真正理解意图。

  这三个特定部分为：
  
  1. 函数名称：如果将函数名重命名为与任务无关的名称（如 "fun"）时，模型就失效。这种失败模式表明，模型既没有理解问题描述，也无法从给出的使用示例中学习到推理模式。即模型严重依赖函数名称，从训练数据集中复制名称相同或相似的片段；
  2. 问题描述中的关键字：移除某些关键词且不降低描述的语义，而且丢失的任何信息都应可从上下文的其余部分恢复，在这种情况下模型失效，说明模型依赖于表面词汇 cues 或在训练过程中看到的频繁出现的术语；
  3. prompt 中提供的示例：如果没有示例，模型就无法生成正确的代码。表明模型需要额外的示例来过滤掉错误的解决方案，作者将这种效果与推理能力差联系起来。
     
![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/7b410a44-108f-475e-ab54-67245aaa547e)





# 四、NLP相关论文
## 4.1 基础论文
- ### [Transformer : Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)  【NIPS 2017】

  :date: 2017
  
  :school: Author Affiliation : Google Brain, Google Research, University of Toronto
  
  主要优势：
  1. 简单，完全基于注意力机制，没有递归和卷积；
  2. 速度快，并行性更高，需要的训练时间也大大减少；
  3. 效果好。
    
- ### [BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

  :date: 2018
  
  :school: Author Affiliation : Google AI Language

  BERT是一种预训练语言模型，它基于Transformer架构，通过在大规模语料库上进行无监督训练，可以实现在各种 NLP 任务上的微调，比如文本分类、问答系统和命名实体识别等。

- ### [GPT-1 : Improving Language Understanding by Generative Pre-Training](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)

  :date: 2018
  
  :school: Author Affiliation : OpenAI

  GPT-1 是一个相当经典的自回归语言模型, 并且他是生成式(Generative)的无监督方式预训练(Pre-Train)模型。GPT-1 是 OpenAI 开发的第一个基于 transformer 架构的大型语言模型。它于2018年6月发布，是当时最大的语言模型之一，使用了 40 亿个参数进行训练。
  
- ### [GPT-2 : Language Models are Unsupervised Multitask Learners](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)

  :date: 2019
  
  :school: Author Affiliation : OpenAI

  GPT-1 的问题 ： fine-tune 只能用到特定任务中，如果 fine-tune 一个分类任务，不能用到句子相似度中去。OPENAI 希望用一个模型做所有 NLP 的任务，尽量做所有的任务。

  GPT2的目标：用一个 LM 去解决任何 NLP 的任务。下游任务是 zero-shot。

  针对GPT-1的问题，GPT-2 作了如下改进： 去掉了 fine-tuning 层：不再针对不同任务分别进行微调，不定义这个模型应该做什么任务，模型会自动识别出来需要做什么任务。在预训练阶段，GPT-2 采用了多任务的方式，不单单只在一个任务上进行学习，而是多个，每一个任务都要保证其损失函数能收敛，不同的任务是共享参数的，这样能进一步的提升模型的泛化能力，因此在即使没有 fine-turning 的情况下依旧有非常不错的表现。

- ### [GPT-3 : Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

  :date: 2020
  
  :school: Author Affiliation : OpenAI

  GPT-3 提出了上下文学习（In Context Learning）的概念，根据上下文包含的任务样本数量，分成了 Zero-Shot（无任务样本）、One-Shot（仅一个任务样本）和 Few-Shot（多个任务样本）三类。

  1. Zero-shot，给出任务描述和提示；
  2. One-shot，给出任务描述、一个示例样本和提示；
  3. Few-shot，给出任务描述、多个示例样本和提示，示例样本数在10至100之间；

  模型在推理阶段，根据上下文进行推理给出输出。
  

