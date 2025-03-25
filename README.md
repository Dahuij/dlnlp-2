# 基于LDA主题模型的金庸小说文本分类

本项目使用LDA（Latent Dirichlet Allocation）主题模型对金庸小说语料库进行文本建模和分类，探究不同参数设置对分类性能的影响。

## 项目结构

- `lda_text_classification.py`: 主要实验代码
- `实验结果分析.md`: 实验方法和结果分析
- `requirements.txt`: 项目依赖包
- `jyxstxtqj_downcc.com/`: 金庸小说语料库

## 实验内容

本实验探究以下三个问题：

1. 在设定不同的主题个数T的情况下，分类性能是否有变化？
2. 以"词"和以"字"为基本单元下分类结果有什么差异？
3. 不同取值的K的短文本和长文本，主题模型性能上是否有差异？

## 运行方式

1. 安装依赖包

```bash
pip install -r requirements.txt
```

2. 运行实验

```bash
python lda_text_classification.py
```

3. 查看结果

实验结果将保存在：
- `lda_classification_results.csv`: 包含所有实验结果的数据表
- `lda_classification_results.png`: 实验结果可视化图表

## 参数设置

- 段落长度K选项: [20, 100, 500, 1000, 3000]
- 主题数量T选项: [5, 10, 20, 50, 100]
- 基本单元选项: ["word", "char"]

## 注意事项

- 实验过程可能需要较长时间，特别是在处理长段落和大量主题的情况下
- 确保有足够的内存处理文本数据
- 金庸小说语料库应放在`jyxstxtqj_downcc.com`目录下 