import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 我们要把模型存到的地方
save_directory = "./minimind/weights"

# 创建目录（如果不存在）
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print("🚀 HuggingFace 网页挂了？没关系，我们用代码自动下载 GPT-2 (作为 MiniMind 的替身)...")
print("⏳ 正在连接服务器，请稍等 (约 100-200MB)...")

try:
    # 自动下载 GPT-2 (最经典的微型模型)
    model_name = "gpt2"

    # 下载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print("✅ Tokenizer 下载并保存成功！")

    # 下载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    print("✅ Model 下载并保存成功！")

    print(f"\n🎉 搞定！模型已保存在: {save_directory}")
    print("我们可以继续 Phase 1.3 了！")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("如果这个也失败，可能是网络完全不通，或者你需要挂个梯子/代理。")