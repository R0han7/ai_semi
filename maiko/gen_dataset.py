import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import csv # CSVを扱うためにインポート

# --- 1. 初期設定（モデルとデータセットの準備）---
# この部分は、初回実行時にダウンロードが走りますが、
# 2回目以降はPCに保存されたキャッシュから高速に読み込まれます。

print("モデルとプロセッサを準備しています...")
# 前回問題があった torch_dtype を削除し、GPUで正しく動くように修正
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16, 
).to("cuda:0")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print("モデルの準備が完了しました。")

print("データセットを準備しています...")
# streaming=Trueにすることで、全データをダウンロードせずに少しずつ使える
fashion_dataset = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)
# データセットから処理したい画像を取得 (ここでは100件)
images_to_process = [item['image'] for item in fashion_dataset.take(100)]
print("データセットの準備が完了しました。")


# --- 2. メイン処理（データセット生成とCSV保存）---
# ここで実際に画像一枚一枚に処理を行い、結果をCSVファイルに保存していきます。

print("データセットの生成を開始します...")
# CSVファイルを開く（なければ新規作成される）
with open('fashion_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # ヘッダー（一番上の行）を書き込む
    writer.writerow(['image_index', 'description'])

    # 各画像に対して説明文を生成し、CSVに書き込む
    for i, image in enumerate(images_to_process):
        # VLMへの指示文（プロンプト）
        prompt = "USER: <image>\nこのファッションアイテムの種類、色、特徴を簡潔に説明してください。\nASSISTANT:"
        
        # モデルに入力するためのデータ準備
        # ここにあった torch_dtype も削除
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

        # 説明文の生成
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 生成されたテキスト部分だけを取り出す
        description = generated_text.split("ASSISTANT:")[1].strip()

        # CSVファイルに結果を一行書き込む
        writer.writerow([f"image_{i+1}", description])

        # 進捗を表示
        print(f"--- 画像 {i+1} / 100 を処理しました ---")
        print(f"説明文: {description}\n")

print("データセットの生成が完了しました。'fashion_dataset.csv' を確認してください。")