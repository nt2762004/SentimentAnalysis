import json
from llama_cpp import Llama

# Tải mô hình DeepSeek Llama
model_path = r"C:\Users\nguye\OneDrive\Desktop\llama.cpp\models\deepseek-llm-7b-chat.Q5_K_M.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=32, n_ctx=2048)

# Prompt tạo dữ liệu positive
prompt = "Write a short English story contains 200 words about a great experience with a product or service."

# Danh sách lưu dữ liệu
data = []

# Sinh dữ liệu positive
num_positive = 500

for i in range(num_positive):
    response = llm(prompt, max_tokens=300, stream=False)

    # Lấy kết quả đầu ra từ mô hình
    generated_text = response["choices"][0]["text"].strip()

    # Lưu dữ liệu vào danh sách
    data.append({"text": generated_text, "label": "positive"})

    print(f"Generated [{i+1}/{num_positive}]: {generated_text[:1000]}...")

# Lưu dữ liệu vào file JSON
output_file = "positive_sentiment_data1.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"\nData saved to {output_file}")