import json
import os

def format_example(ex):
    """
    JSONLの1行を学習用のテキストフォーマットに変換する。
    思考プロセス（JSON部分）を先に出力させ、最後にメッセージを出力させる。
    """
    # ユーザーの入力を模倣（stateの情報から再構成）
    problem = ex["state"].get("problem", "ビジネス相談")
    metrics = ex["state"].get("metrics", {})
    known = ex["state"].get("known_info", [])
    
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items() if v is not None])
    known_str = "、".join(known)
    
    input_text = f"相談内容: {problem}\n"
    if metrics_str:
        input_text += f"現状の指標: {metrics_str}\n"
    if known_str:
        input_text += f"既知の事実: {known_str}\n"
    
    # モデルの出力を構成
    # 思考と判断のJSON構造
    analysis_part = {
        "state": ex["state"],
        "analysis": ex["analysis"],
        "decision": ex["decision"],
        "reopen_conditions": ex["reopen_conditions"]
    }
    
    response_json = json.dumps(analysis_part, ensure_ascii=False, indent=2)
    
    # 最終的なプロンプト
    prompt = f"### ユーザーの相談:\n{input_text}\n\n### 懐疑コアの内部分析と判断:\n{response_json}\n\n### 回答メッセージ:\n{ex['message']}"
    
    return {"text": prompt}

def main():
    input_file = "dataset.jsonl"
    output_file = "train_data.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    formatted_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                formatted_data.append(format_example(ex))
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Successfully converted {len(formatted_data)} examples to {output_file}")

if __name__ == "__main__":
    main()
