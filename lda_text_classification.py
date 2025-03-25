import os
import re
import random
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim import corpora, models

# 设置随机种子以确保结果可重现
random.seed(42)
np.random.seed(42)

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义语料库路径和段落长度选项
corpus_dir = "jyxstxtqj_downcc.com"
K_options = [20, 100, 500, 1000, 3000]  # 段落长度选项
T_options = [5, 10, 20, 50, 100]  # 主题数量选项
unit_options = ["word", "char"]  # 基本单元选项：词或字

# 定义结果保存路径
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_corpus():
    """加载金庸小说语料库"""
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt') and file_name != "inf.txt":
            novel_name = file_name[:-4]
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='gb18030') as f:
                text = f.read().replace('\n', '').replace(' ', '')
                # 移除非中文字符和标点符号
                text = re.sub(r'[^\u4e00-\u9fff。，！？：；""''（）【】《》、]', '', text)
            novels.append((novel_name, text))
    return novels

def preprocess_text(text, unit="word"):
    """对文本进行分词或以字为单位切分"""
    if unit == "word":
        return list(jieba.cut(text))
    elif unit == "char":
        return list(text)

def sample_paragraphs(novels, k, n_samples=1000):
    """均匀抽取段落"""
    paragraphs, labels = [], []
    for novel_name, text in novels:
        tokens = preprocess_text(text, unit="char")  # 先按字符切分
        num_paragraphs = len(tokens) // k
        sampled = [tokens[i * k:(i + 1) * k] for i in range(num_paragraphs)]
        sampled = random.sample(sampled, min(len(sampled), n_samples // len(novels)))
        paragraphs.extend(sampled)
        labels.extend([novel_name] * len(sampled))
    return paragraphs, labels

def train_lda(paragraphs, num_topics):
    """训练LDA模型"""
    dictionary = corpora.Dictionary(paragraphs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # 过滤低频词
    corpus = [dictionary.doc2bow(text) for text in paragraphs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, 
                              passes=20,  # 减少迭代次数
                              alpha='auto', 
                              eta='auto')
    return lda_model, dictionary, corpus

def transform_paragraphs(lda_model, dictionary, paragraphs):
    """将段落转换为主题分布"""
    corpus = [dictionary.doc2bow(text) for text in paragraphs]
    topic_distributions = []
    for doc in lda_model[corpus]:
        topic_vector = [0] * lda_model.num_topics
        for topic_id, prob in doc:
            topic_vector[topic_id] = prob
        topic_distributions.append(topic_vector)
    return np.array(topic_distributions)

def run_classification_experiment(topic_distributions, labels):
    """运行分类实验"""
    classifier = RandomForestClassifier(
        n_estimators=50,  
        max_depth=8,     
        random_state=42,
        n_jobs=4          
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 
    accuracies = []
    
    for train_idx, test_idx in kf.split(topic_distributions):
        X_train, X_test = topic_distributions[train_idx], topic_distributions[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    return np.mean(accuracies), np.std(accuracies)

def run_experiments():
    print("开始读取语料库...")
    novels = load_corpus()
    print(f"读取了{len(novels)}部小说，准备进行实验...")
    
    results = []
    
    for unit in unit_options:
        for k in K_options:
            print(f"\n测试基本单元: {unit}, 段落长度 K = {k}")
            paragraphs, labels = sample_paragraphs(novels, k, n_samples=500)  # 减少样本数量
            print(f"从{len(novels)}部小说中均匀抽取了{len(paragraphs)}个段落")
            
            for t in T_options:
                print(f"\n测试主题数 T = {t}")
                print("正在进行特征提取和主题建模...")
                
                # 训练LDA模型
                lda_model, dictionary, _ = train_lda(paragraphs, t)
                topic_distributions = transform_paragraphs(lda_model, dictionary, paragraphs)
                
                # 运行分类实验
                accuracy, std = run_classification_experiment(topic_distributions, labels)
                results.append({
                    "实验": "组合效应",
                    "段落长度(K)": k,
                    "主题数(T)": t,
                    "基本单元": unit,
                    "准确率": accuracy,
                    "标准差": std
                })
                print(f"平均准确率: {accuracy:.4f} ± {std:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "lda_classification_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n实验结果已保存到 {csv_path}")
    
    # 绘制结果图表
    plot_results(results)

def plot_results(results):
    df = pd.DataFrame(results)
    
    # 绘制图1：不同段落长度下的主题数量影响
    plt.figure(figsize=(15, 10))
    for i, k in enumerate(K_options):
        plt.subplot(2, 3, i+1)
        subset = df[(df["段落长度(K)"] == k)]
        
        # 绘制词单元的结果
        word_results = subset[subset["基本单元"] == "word"]
        plt.errorbar(word_results["主题数(T)"], word_results["准确率"], 
                    yerr=word_results["标准差"], marker='o', label='词单元')
        
        # 绘制字单元的结果
        char_results = subset[subset["基本单元"] == "char"]
        plt.errorbar(char_results["主题数(T)"], char_results["准确率"], 
                    yerr=char_results["标准差"], marker='s', label='字单元')
        
        plt.xlabel("主题数量 (T)")
        plt.ylabel("准确率")
        plt.title(f"段落长度 K={k} 的主题数量影响")
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "lda_classification_results.png")
    plt.savefig(png_path)
    print(f"实验结果图表已保存到 {png_path}")

if __name__ == "__main__":
    run_experiments() 