import os
import json
import time
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import datetime
import re

# =============================================================================
# セクション 1: Gemini APIとの通信を担当するコア関数
# =============================================================================

def initialize_gemini_model(api_key, model_name):
    """
    APIキーを設定し、指定されたモデルのGenerativeModelインスタンスを初期化する。
    これが現在推奨される、最も安定した初期化方法。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        print(f"Gemini model '{model_name}' initialized and validated successfully.")
        return model
    except Exception as e:
        print(f"Fatal: Error initializing Gemini model. Please check API key and model name. Error: {e}")
        return None

def get_gemini_response(model, prompt):
    """
    モデルインスタンスとプロンプトを受け取り、APIを呼び出して結果を返す。
    JSONのパースとエラーハンドリングもここで行う。
    """
    raw_response_text = "ERROR: NO_RESPONSE"
    try:
        # JSON出力を強制するための設定
        generation_config = GenerationConfig(response_mime_type="application/json")
        
        # API呼び出し
        response = model.generate_content(contents=prompt, generation_config=generation_config)
        raw_response_text = response.text
        
        # LLMが返しがちなMarkdownブロックを正規表現で除去
        match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", raw_response_text)
        json_str = match.group(2).strip() if match else raw_response_text.strip()
        
        # JSONをパースし、'label'キーの値を取得
        result_json = json.loads(json_str)
        parsed_label = result_json.get('label', 'ERROR_MISSING_KEY')
        return parsed_label, raw_response_text
    
    except json.JSONDecodeError:
        return "ERROR_JSON_DECODE", raw_response_text
    except Exception as e:
        # APIからのブロックなど、その他のエラー
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
             print(f"\nAPI Error. Prompt Feedback: {response.prompt_feedback}")
        else:
             print(f"\nAn unexpected API error occurred: {e}")
        return "ERROR_API", raw_response_text

# =============================================================================
# セクション 2: 評価の実行と結果の分析・保存を担当する関数
# =============================================================================

def run_evaluation_for_prompt(model, config, test_df, prompt_obj, prompt_name, output_dir):
    """単一のプロンプトでデータセット全体を評価し、ログと結果を返す"""
    print(f"\n===== Evaluating with prompt: '{prompt_name}' =====")
    
    # prompt_template = prompt_obj['template']
    prompt_template = """あなたは対話分析の専門家です。以下の例を参考に、「ユーザーの発言」と「それに対する応答」を分析し、応答が皮肉(Irony)か、そうでない(Not Irony)かを判断してください。

# 指示
- 応答が皮肉を含んでいる場合は "IRONY" と分類してください。
- 応答が皮肉を含んでいない場合は "NOTIRONY" と分類してください。
- 回答は必ず `{{\"label\": \"分類結果\"}}` というJSON形式で出力してください。

---
# 例1
ユーザーの発言: このレストラン、すごく評価が高いんだって！
応答: うん、この生温かいスープとゴムみたいなステーキは、まさに至高の味だね。
# JSON出力
{{\"label\": \"IRONY\"}}
---
# 例2
ユーザーの発言: 今日のプレゼンどうだった？
応答: すごく分かりやすかったよ！特にデータの説明が丁寧で助かった。
# JSON出力
{{\"label\": \"NOTIRONY\"}}
---

# 本番の対話
ユーザーの発言: {utterance}
応答: {response}
# JSON出力
"""
    print(f"Prompt template content: {prompt_template}")
    delay = config['api_settings']['request_delay_seconds']
    references = test_df['label'].map({0: "NOTIRONY", 1: "IRONY"}).tolist()
    
    log_entries = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Prompt: {prompt_name}"):
        
        utterance_val = str(row['Utterance'])
        response_val = str(row['Response'])
        prompt = prompt_template.format(utterance=utterance_val, response=response_val)
        print(f"\nProcessing index {index}: {prompt}")
        parsed_label, raw_response = get_gemini_response(model, prompt)
        
        # エラー発生時にコンソールに通知
        if "ERROR" in parsed_label:
            print(f"\n[!] Warning at index {index}: Parsed as '{parsed_label}'. Raw response: '{raw_response}'")

        log_entries.append({
            'index': index, 'reference_label': references[index], 'predicted_label': parsed_label,
            'utterance': row['Utterance'], 'response': row['Response'],
            'llm_response_raw': raw_response, 'prompt': prompt
        })
        time.sleep(delay)

    # 詳細ログをCSVに保存
    log_df = pd.DataFrame(log_entries)
    log_path = os.path.join(output_dir, f"predictions_log_{prompt_name}.csv")
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"Detailed prediction log saved to: {log_path}")

    # 性能指標を計算して返す
    predictions = log_df['predicted_label'].tolist()
    return analyze_and_save_report(references, predictions, prompt_name, output_dir)

def analyze_and_save_report(references, predictions, prompt_name, output_dir):
    """予測結果からレポートと混同行列を生成・保存し、サマリーを返す"""
    # エラーラベルも含めてレポートを作成
    labels_for_report = sorted(list(set(references) | set(predictions)))
    
    report_text = classification_report(references, predictions, labels=labels_for_report, zero_division=0)
    report_dict = classification_report(references, predictions, labels=labels_for_report, output_dict=True, zero_division=0)
    
    print("\n--- Classification Report ---")
    print(report_text)
    
    # 混同行列の描画
    cm = confusion_matrix(references, predictions, labels=labels_for_report)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_for_report)
    fig, ax = plt.subplots(figsize=(len(labels_for_report)*2, len(labels_for_report)*2))
    disp.plot(cmap="Blues", ax=ax, colorbar=False, xticks_rotation='vertical')
    ax.set_title(f"Confusion Matrix\nPrompt: {prompt_name}")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"cm_{prompt_name}.png")
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")
    
    # サマリー用の結果を作成
    error_count = sum(1 for p in predictions if "ERROR" in p)
    summary = {
        "prompt_name": prompt_name,
        "accuracy": report_dict.get('accuracy', 0),
        "f1_score_IRONY": report_dict.get('IRONY', {}).get('f1-score', 0),
        "f1_score_NOTIRONY": report_dict.get('NOTIRONY', {}).get('f1-score', 0),
        "f1_score_macro": report_dict.get('macro avg', {}).get('f1-score', 0),
        "error_count": error_count
    }
    return summary

# =============================================================================
# セクション 3: 全体の処理を統括するmain関数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on irony detection based on a config file.")
    parser.add_argument("--config", type=str, default="config.llm.json", help="Path to the JSON config file.")
    parser.add_argument("--api_key", type=str, default=None, help="Your Google API Key (overrides environment variable).")
    args = parser.parse_args()

    # 1. 設定ファイルの読み込み
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuration loaded from '{args.config}'")
    except Exception as e:
        print(f"Error: Could not load or parse config file '{args.config}'. Error: {e}")
        return

    # 2. APIキーの準備とモデルの初期化
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please provide it via --api_key argument or set GOOGLE_API_KEY.")
    
    model = initialize_gemini_model(api_key, config['api_settings']['model_name'])
    if not model:
        return

    # 3. 出力ディレクトリの準備
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_base = config['evaluation_settings']['output_dir_base']
    output_dir = f"{output_dir_base}_{config['api_settings']['model_name'].replace('models/', '')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nAll evaluation outputs will be saved to: {output_dir}")

    # 4. 評価の実行ループ
    all_results_summary = []
    for dataset_name, csv_path in config['evaluation_settings']['datasets_to_test'].items():
        if not os.path.exists(csv_path):
            print(f"\nWarning: Dataset file '{csv_path}' for '{dataset_name}' not found. Skipping.")
            continue
        
        print(f"\n\n<<<<< Processing Dataset: {dataset_name.upper()} ({csv_path}) >>>>>")
        test_df = pd.read_csv(csv_path)
        print(f"Columns in {csv_path}: {test_df.columns.tolist()}")

        for prompt_name, prompt_obj in config['prompt_templates'].items():
            prompt_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(prompt_output_dir, exist_ok=True)
            
            summary = run_evaluation_for_prompt(model, config, test_df, prompt_obj, prompt_name, prompt_output_dir)
            summary['dataset_name'] = dataset_name
            all_results_summary.append(summary)

    # 5. 最終サマリーの表示と保存
    if all_results_summary:
        results_df = pd.DataFrame(all_results_summary)
        print("\n\n===== FINAL SUMMARY OF ALL RUNS =====")
        display_cols = ['dataset_name', 'prompt_name', 'accuracy', 'f1_score_macro', 'f1_score_IRONY', 'f1_score_NOTIRONY', 'error_count']
        print(results_df[display_cols].round(4))
        
        summary_path = os.path.join(output_dir, "summary_of_all_runs.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\nFinal summary saved to {summary_path}")
    else:
        print("\nNo datasets were evaluated.")

if __name__ == "__main__":
    main()
