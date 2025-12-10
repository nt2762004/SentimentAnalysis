import json
from llama_cpp import Llama

# Tải mô hình DeepSeek Llama
model_path = r"C:\Users\nguye\OneDrive\Desktop\llama.cpp\models\deepseek-llm-7b-chat.Q5_K_M.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=32, n_ctx=2024)

# Prompt tạo dữ liệu negative
prompt = "Write a short English story contains 200 words about a disappointing experience with a product or service."

# Danh sách lưu dữ liệu
data = []

num_negative = 1

for i in range(num_negative):  # Sinh 1000 văn bản negative
    response = llm(prompt, max_tokens=300, stream=False)

    # Lấy kết quả đầu ra từ mô hình
    generated_text = response["choices"][0]["text"].strip()

    # Lưu dữ liệu vào danh sách
    if len(generated_text.split()) >= 20:  # Chỉ lưu nếu có ít nhất 10 từ
        data.append({"text": generated_text, "label": "negative"})


    print(f"Generated [{i+1}/{num_negative}]: {generated_text[:100]}...")  # Hiển thị 100 ký tự đầu để kiểm tra

# Lưu dữ liệu vào file JSON
output_file = "negative_sentiment_data500.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"\nData saved to {output_file}")
