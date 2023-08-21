# papers_of_codegen
Some paper about code generation.

If you have any papers you'd like to recommend, you can contact me at 839808679@qq.com



# 大模型汇总（LLMs Summary）



# 近期有创意的论文（Recent Innovative Papers）
## 模型水印
- ### [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226v3.pdf)  【ICML 2023】

  Tue, 24 Jan 2023
  
  大型语言模型(LLMs)的发展促进了人类对工具的使用，但是也会导致更多的恶意攻击，如利用AI操控选举活动，制造假新闻，制造假数据，以及在学术写作和编码作业中作弊。让LLMs生成的文本有水印，可以极大程度的避免这些问题。
  
  作者提出了一种有效的水印，可以在短跨度的token（少至 25 个token）中检测到合成文本，而误报（人类文本被标记为机器生成）在统计上是不可能的。水印检测算法可以被公开，使第三方（例如社交媒体平台）能够自行运行，也可以保持私有，并在 API 上运行。

  该水印有以下特性：
  - 1、水印检测不需要通过调用  LLM API 或者获知 LLM 参数就可以实现模型
  - 2、不需要额外训练就以对文本添加水印
  - 3、即便只有很小的一段生成文本，水印也可以被检测
  - 4、除非大幅修改生成文本，水印无法被移除
  - 5、针对水印检测，可以计算出一个严格的统计量
  - 6、水印对人类肉眼来说是不可见的

  作者提出的两种水印方法：hard watermarking 和 a more sophisticated watermark。

  hard watermarking 总体思路是，将上一个词的哈希值作为 seed，使用这个 seed，将词汇表随机划分为大小相等的“绿名单”G 和“红名单”R，即，词汇表中的任意一个词都会被随机到 G 或 R。下一个生成的词，在 sample 的时候只会去绿名单中sample。但对于某些固定搭配，基本上只会以 AB 的形式出现，一旦 B 存在于红名单中，A 后面只能搭配绿名单的词，无论选什么词都会使得单词搭配出现问题。作者提出的第二种方法相当于“建议”它选择率列表中的词，而不是强硬的规定。


  
- ### [Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/pdf/2305.15060.pdf)

  Wed, 24 May 2023

  作者发现，以往对于文本的水印生成方法（如分成两组token，优先选择绿色token），对于代码生成任务来说不管用，因为模型输出的质量会受到严重影响。
  
  在这项工作中，作者提出了一种新的水印方法 —— SWEET（Selective WatErmarking via Entropy Thresholding），作者不对代码输出中的每个token应用绿-红规则，而是只选择性地将水印应用于熵值足够高、超过规定阈值的 token。这种选择性水印方法不会对 gold tokens 应用绿-红规则，以防止它们落入红色组，因此不会出现上述问题。







# 评测相关论文（Papers Related to Review）


# 基础论文
