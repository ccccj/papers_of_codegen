# papers_of_codegen [![My github](https://skillicons.dev/icons?i=github)](https://github.com/ccccj/papers_of_codegen) 

Some paper about code generation.

> If you have any papers you'd like to recommend, you can contact me at :envelope: 839808679@qq.com

- [papers\_of\_codegen ](#papers_of_codegen-)
- [一、大模型汇总（LLMs Summary）](#一大模型汇总llms-summary)
  - [1.1 开源模型](#11-开源模型)
  - [1.2 闭源模型](#12-闭源模型)
  - [1.3 表格汇总](#13-表格汇总)
- [二、Code 论文（Papers of Code）](#二code-论文papers-of-code)
  - [2.1 模型水印](#21-模型水印)
  - [2.2 code debug](#22-code-debug)
  - [2.3 tokenization](#23-tokenization)
  - [2.4 code decoding](#24-code-decoding)
  - [2.5 Code 辅助信息](#25-code-辅助信息)
  - [2.6 code representation](#26-code-representation)
- [三、评测相关论文（Papers Related to Evaluation）](#三评测相关论文papers-related-to-evaluation)
  - [3.1 评测框架](#31-评测框架)
- [四、NLP相关论文](#四nlp相关论文)
  - [4.1 基础论文](#41-基础论文)
  - [4.2 调参方法](#42-调参方法)
  - [4.3 数据 sample 方法](#43-数据-sample-方法)

# 一、大模型汇总（LLMs Summary）
## 1.1 开源模型
- ### [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf)

  :date: Sat, 13 May 2023

  :school: Author Affiliation : Salesforce AI Research

  :open_file_folder: Model Size: 220M, 770M, 2B, 6B, 16B

- ### [PolyCoder: A SYSTEMATIC EVALUATION OF LARGE LANGUAGE MODELS OF CODE](https://arxiv.org/pdf/2202.13169.pdf)
  
  :date: Sat, 26 Feb 2022

  :school: Author Affiliation : Carnegie Mellon University

  :open_file_folder: Model Size: 160M, 400M, 2.7B

  作者对跨不同编程语言的现有 code 模型——Codex、GPT-J、GPT-Neo、GPT-NeoX 和 CodeParrot 进行了系统评估。他们希望通过比较这些模型来进一步了解代码建模设计决策的前景，并指出关键的缺失一环，即迄今为止，没有大规模开源语言模型专门针对**多编程语言的代码**进行训练。
  作者发现，不同语言的语料会对彼此有提示作用，用单一语言预料训练的模型（如CodeParrot），即使增大参数量，效果也不提升，侧面印证了上面的结论。

  同时，作者推出了三个此类模型，参数量从 160M 到 2.7B，命名为「PolyCoder」，并开源了该模型和数据集。
  
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
  
- ### [phi-1: Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644.pdf)

  :date: Tue, 20 Jun 2023
  
  :school: Author Affiliation : Microsoft Research

  :open_file_folder: Model Size: 1.3B

  phi-1，规模明显小于其他同类模型，拥有 1.3B 参数。尽管规模很小，但 phi-1 在 HumanEval 和 MBPP 中的 pass@1 分别达到了 50.6% 和 55.5%。作者在这篇论文中主要使用的方法就是提高数据质量。作者认为，大家普遍使用的一些数据源并不是教模型学习如何推理和规划算法的最佳来源，所以作者主要讲述了如何整理数据。

  在论文中，作者主要用到的三个数据集：
  1. 一个经过过滤的 code 数据集，它是 The Stack 和 StackOverflow 的子集，通过分类器获得（6B大小）（由 GPT-4 对 code 生成注释得到的数据，来训练该分类器）；
  2. 合成的教科书数据集，小于1B，由 GPT-3.5 生成的 Python 教科书组成（作者在该数据集中引入了随机性，但具体 prompt 没说）；
  3. 一个小型的合成练习题数据集，由  Python 练习题和答案组成，大约180M，也是由 GPT-3.5 生成的（让GPT-3.5生成样例的提示工程不清楚）。

  为了排除 CodeExercises 数据集的污染，造成 HumanEval 评分很高的因素，作者的团队以与 HumanEval 相同的格式创建了 50 个不太可能出现在现实世界代码库或编码练习中的新问题，并用 GPT-4 给解决方案打分的。另外还删除与 HumanEval 中 "相似 "的文件来修剪 CodeExercises 数据集，发现重新训练后的 phi-1 仍然优于 StarCoder。。
  
## 1.3 表格汇总

|Model|Model Size|HumanEval pass@1|HumanEval pass@10|HumanEval pass@100|MBPP pass@1|public|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LaMDA|137B|14.0|-|47.3|-|:white_large_square:|
|AlphaCode|1.1B|17.1|28.2|45.3|-|:white_large_square:|
|MIM|1.3B|22.4|41.7|53.8|-|:white_large_square:|
|MIM|2.7B|30.7|48.2|69.6|-|:white_large_square:|
|PaLM|8B|3.6|-|18.7|-|:white_large_square:|
|PaLM|62B|15.9|-|46.3|-|:white_large_square:|
|PaLM|540B|26.2|-|76.2|-|:white_large_square:|
|PaLM-Coder|540B|36.0|-|88.4|47.0|:white_large_square:|
|Codex|2.5B|21.4|35.4|59.5|-|:white_large_square:|
|Codex|12B|28.8|46.8|72.3|-|:white_large_square:|
|code-cushman-001|-|33.5|54.3|77.4|-|:white_large_square:|
|code-davinci-002|-|47.0|74.9|92.1|-|:white_large_square:|
|GPT-3.5|-|48.1|-|-|-|:white_large_square:|
|GPT-4|-|67.0|-|-|-|:white_large_square:|
|GPT-Neo|2.7B|6.4|11.3|21.4|-|:white_check_mark:|
|GPT-J|6B|11.6|15.7|27.7|-|:white_check_mark:|
|GPT-NeoX|20B|15.4|25.6|41.2|-|:white_check_mark:|
|InCoder|1.3B|8.9|16.7|25.6|-|:white_check_mark:|
|InCoder|6B|15.2|27.8|47.0|-|:white_check_mark:|
|CodeGeeX|13B|22.9|39.6|60.9|24.4|:white_check_mark:|
|LLaMA|7B|10.5|-|36.5|-|:white_check_mark:|
|LLaMA|13B|15.8|-|52.5|-|:white_check_mark:|
|LLaMA|33B|21.7|-|70.7|-|:white_check_mark:|
|LLaMA|65B|23.7|-|79.3|-|:white_check_mark:|
|Replit|3B|21.9|-|-|-|:white_check_mark:|
|StarCoder|15B|33.6|-|-|52.7|:white_check_mark:|
|CodeGen-mono|2B|23.7|36.6|57.0|-|:white_check_mark:|
|CodeGen-mono|6B|26.1|42.3|65.8|-|:white_check_mark:|
|CodeGen-mono|16B|29.3|49.9|75.0|-|:white_check_mark:|
|CodeT5+|220M|12.0|20.7|31.6|-|:white_check_mark:|
|CodeT5+|770M|15.5|27.2|42.7|-|:white_check_mark:|
|CodeT5+|2B|24.2|38.2|57.8|-|:white_check_mark:|
|CodeT5+|6B|28.0|47.2|69.8|-|:white_check_mark:|
|CodeT5+|16B|30.9|51.6|76.7|-|:white_check_mark:|
|InstructCodeT5+|16B|35.0|54.5|77.9|-|:white_check_mark:|
|StarCoder (prompted)|15B|40.8|-|-|49.5|:white_check_mark:|
|CodeGen-mono w/ CodeT|16B|36.7|59.3|-|-|:white_check_mark:|
|CodeT5+ w/ CodeT|16B|38.5|63.6|77.1|-|:white_check_mark:|
|InstructCodeT5+ w/ CodeT|16B|42.9|67.8|78.7|-|:white_check_mark:|


# 二、Code 论文（Papers of Code）
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

  :school: Author Affiliation : Seoul National University, NAVER AI Lab, NAVER Cloud

  作者发现，以往对于文本的水印生成方法（如分成两组token，优先选择绿色token），对于代码生成任务来说不管用，因为模型输出的质量会受到严重影响。
  
  在这项工作中，作者提出了一种新的水印方法 —— SWEET（Selective WatErmarking via Entropy Thresholding），作者不对代码输出中的每个token应用绿-红规则，而是只选择性地将水印应用于熵值足够高、超过规定阈值的 token。这种选择性水印方法不会对 gold tokens 应用绿-红规则，以防止它们落入红色组，因此不会出现上述问题。

## 2.2 code debug
- ### [Teaching Large Language Models to Self-Debug](https://arxiv.org/pdf/2304.05128.pdf)
  
  :date: Tue, 11 Apr 2023

  :school: Author Affiliation : Google Research, UC Berkeley

  作者证明了 Self-Debug 可以教会大型语言模型执行小黄鸭调试法————在程序的调试或测试过程中，写代码的人耐心地向小黄鸭解释每一行程序的作用，以此来激发灵感与发现矛盾，即，自言自语解释程序来找到 bug。也就是在没有任何关于代码正确性或错误消息的反馈的情况下，模型能够通过用自然语言解释来识别代码中的错误。具体步骤如下：
  1. 模型生成新代码，然后执行代码并让模型解释代码（当没有单元测试时，反馈信息可以纯粹基于代码解释）;
  2. 代码解释和执行结果构成反馈信息，然后反馈信息被发送回模型以执行更多的调试步骤。

![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/84a9d187-1420-4f66-b87b-b94ec7bbb0f0)

  在作者的实验中，作者使用用预训练的大型语言模型（code-davinci-002），无需进行微调。根据问题描述，模型首先预测候选程序，然后推断程序的正确性，并为后续调试步骤生成反馈信息。当反馈信息表明预测正确或达到允许的最大调试次数时，调试过程结束。作者设计了三种不同的反馈信息：
  1. SELF-DEBUGGING with Simple Feedback
  2. SELF-DEBUGGING with Unit Tests (UT)
  3. SELF-DEBUGGING via Code Explanation (Expl.)

  在没有单元测试来验证预测正确性的 Spider benchmark 上，带有代码解释的自调试持续将 baseline 提高了 2−3%，并将最难标签问题的预测精度提高了 9%。在可进行单元测试的 TransCoder 和 MBPP 上，自调试可将基线准确度提高高达 12%。同时，通过利用反馈消息和重用失败的预测，SELFDEBUGGING 显着提高了样本效率，并且可以匹配或优于生成超过 10 倍候选程序的 baseline 模型。
  
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
 
## 2.4 code decoding
- ### [Planning with Large Language Models for Code Generation](https://arxiv.org/pdf/2303.05510.pdf)  【ICLR 2023】

  :date: Thu, 9 Mar 2023
  
  :school: Author Affiliation : MIT-IBM Watson AI Lab, The University of Hong Kong, MIT BCS, CBMM, CSAIL, UMass Amherst
  
  现有的大型语言模型在解码过程中常常使用 beam search 或 sample。尽管在基于匹配的评测标准中获得了高分，但生成的代码经常无法编译或生成不正确的输出。主要原因是传统的 decoding 算法可能不是代码生成的最佳选择。在这项工作中，作者提出了一种新颖的 decoding 算法，即 Planning-Guided Transformer Decoding（PG-TD），它使用树搜索算法进行前向搜索并引导 Transformer 生成更好的程序。

  具体来说，Transformer 不是简单地优化生成序列的可能性，而是利用规划器来生成候选程序并在公共测试用例上对其进行测试。作者还设计了一种在 Transformer 和规划器之间共享信息的机制，以使算法具有计算效率。

  虽然作者实验中性能确实不错，但这种方法只适用于提供了测试样例的情景（如算法竞赛、oj等等）。

## 2.5 Code 辅助信息
- ### [Repository-Level Prompt Generation for Large Language Models of Code](https://arxiv.org/pdf/2206.12839.pdf)  【ICML 2023】

  :date: Sun, 26 Jun 2022
  
  :school: Author Affiliation : Mila, Université de Montréal, Google, CIFAR Associate Fellow, McGill University

  这篇论文研究的场景是，类似于 copilot 那样的，用户在输入，模型去预测用户后面可能想输入什么内容。

  作者认为，模型给用户的提示中的相关上下文不仅可以来自当前文件，还可以来自外部，如 import、父类、同一目录下的文件和 API 文档。此外，根据具体情况，相关上下文可能分散在多个位置。由于 LLM 可用于提示的上下文长度有限，选择相关上下文变得越来越重要。作者提出了一个名为 Repo-Level Prompt Generator （RLPG）的框架来解决这一问题。

  ![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/0cc85106-08c1-4026-96d5-ce326a9b5817)

  给定 **prompt 建议列表**、**光标位置**、**关联的存储库**作为输入，prompt 建议分类器（Prompt Proposal Classifier）（PPC）会预测 prompt 建议。其中，PPC 是一个训练好的模型。之后，Prompt Composer 将所选提示提案中的上下文（由 PPC 给出）与 Codex 通常使用的上下文（默认 Codex 上下文）相结合来生成提示。  

## 2.6 code representation


# 三、评测相关论文（Papers Related to Evaluation）
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


- ### [CodeScore: Evaluating Code Generation by Learning Code Execution](https://arxiv.org/pdf/2301.09043.pdf)

  :date: Sun, 22 Jan 2023
  
  :school: Author Affiliation :  Peking University

  作者认为以往的代码评价指标（CEM）有很多缺点。以往的代码评价指标（CEM）可分为两类：
  1. match-based CEMs（基于匹配的CEM），只衡量表面形式的差异，而不管代码的功能等价性；
  2. execution-based CEMs（基于执行的 CEM），有巨大的执行开销，如，收集昂贵的测试用例、解决繁琐的执行依赖关系和巨大的执行时间；其次，会产生安全问题。
 
  于是作者提出了的新的 CEM：CodeScore，可以在不执行代码的情况下，估算生成代码在测试用例上的通过率；既精准，又不需要执行，所以开销小。作者训练了一个模型来估算这个 score：
  ![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/53779483-cbb0-4ab9-8017-2946b57d24e1)

  
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
  
## 4.2 调参方法
- ### [LOMO：Full Parameter Fine-Tuning for large language models with limited resources](https://arxiv.org/pdf/2306.09782.pdf)

  :date: Fri, 16 Jun 2023
  
  :school: Author Affiliation : Fudan University

  最近，高效的参数微调方法，例如 LoRA和 Prefix-tuning，为 LLM 微调提供了解决方案。然而，这些方法并没有为全参数微调提供实用的解决方案，而全参数微调被认为是比参数高效微调更强大的方法。这项工作的目标是探索在资源有限的场景中完成全参数微调技术。

  作者通过各种分析，发现 SGD 在LLM场景下的全参数微调是很有前景的。于是作者进一步的改进 SGD，最终版本称之为 LOMO，可以在 8 块 3090 上微调 65B。与标准方法（DeepSpeed 解决方案）相比，将内存使用量减少到 10.8%。因此，可以在具有 8×RTX 3090（每个具有 24GB 内存）的单台机器上对 65B 模型进行全参数微调。LOMO 技术导致的内存使用量等同于参数加激活和最大梯度张量的使用量，将全参数微调的内存使用量推向了一个极端，使其仅仅相当于推理的使用量。

  作者主要是讨论了 SGD 的一些缺点，并分析这些缺点在 LLM 微调时，不会产生影响，因此可以使用非常节省显存的 SGD 来微调。并且进一步改进了 SGD，使得显存可以更加节省，为全参微调 LLM 提供了方法。


## 4.3 数据 sample 方法
- ### [UNIMAX: FAIRER AND MORE EFFECTIVE LANGUAGE SAMPLING FOR LARGE-SCALE MULTILINGUAL PRETRAINING](https://arxiv.org/pdf/2304.09151.pdf)

  :date: Tue, 18 Apr 2023
  
  :school: Author Affiliation : Google Research

  LLM的训练数据集中通常涉及多种语言，由于不同语言的数据可用性差异很大，因此多语言预训练可以被描述为数据严重不平衡的多任务学习。通常，英语是资源最高的语言，其规模比资源较低的语言大几个数量级。因此，设计此类模型的一个关键问题是“语言平衡”问题：应该以什么比例平衡预训练语言数据集?

  作者提出了一个名为 UniMax 的新型语言采样方法，用于大规模多语言预训练模型。该方法能够更公平地分配语言样本，并且能够在明确限制每种语言语料库的重复次数的情况下，减轻对较少语言的过拟合问题。作者进行了一系列实验，发现 UniMax 在各种多语言基准测试中都优于标准的温度采样方法，并且随着模型规模的增加，其好处依然存在。本文还提供了一个更新版本的 mC4 多语言语料库，由 107 种语言的 29 万亿个字符组成，并公开了复现实验的完整代码。

  UNIMAX 采样的第一步是根据训练语料库中的字符数对语言进行排序。然后从字符数最少的语言开始迭代这些语言。每次迭代时，都会检查剩余的字符预算，看看是否可以在不使用超过 N 个 epochs 的任何语言的情况下将其平均分配给其他语言。如果预算可以平均分配，则在各语言之间统一分配。如果没有，则为语言 l 分配价值 N 个 epochs 的字符，剩余的预算就会减少。

  ![image](https://github.com/ccccj/papers_of_codegen/assets/33253635/41010778-1ad4-4fe9-aeb3-75a7e6a2c786)



  
