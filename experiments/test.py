from transformers import AutoTokenizer

# 加载预训练模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/geshuting/Code/CTAL-main/CTAL-main/chinese-roberta-wwm-ext")

# 输入文本
text = "我喜欢自然语言处理。"

# 使用 tokenizer 进行分词
tokens = tokenizer.tokenize(text)

# 输出分词结果
print(tokens)