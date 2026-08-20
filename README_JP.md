![Needle](assets/banner.png)

# Needle 2 日本語対応ガイド

Needle 2 は、ツール呼び出し・デバイス操作・構造化抽出向けの 4,500 万パラメータモデルです。本リポジトリには、推論、LoRA ファインチューニング、`.cact` 形式への書き出しが含まれています。

このガイドでは、DeepSeek API を使って日本語の教師データを作り、そのデータで Needle を LoRA 学習する流れを説明します。

> 重要: DeepSeek API で DeepSeek 自体を再学習するのではありません。DeepSeek は日本語の学習例を生成・拡張する教師データ作成と評価に使い、実際の学習は Needle のローカル LoRA パイプラインで行います。

## 1. インストール

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e .
```

GPU を使う場合は、環境に合わせて次のいずれかを追加します。

```sh
pip install "cactus-needle[gpu]"    # NVIDIA CUDA
pip install "cactus-needle[metal]"  # Apple Silicon
```

GPU がない環境でも学習できます。Needle は 45M パラメータの LoRA 学習なので小規模データなら CPU で動きますが、GPU よりかなり遅くなります。CPU ではまず `--num-samples 50`、`--epochs 2`、`--max-len 256`、`--batch-size 1`〜`4` で動作確認し、本番データを増やすのがおすすめです。数百例以上・epoch 10 以上・系列長 1024 の学習は、Colab の T4 など GPU を使う方が現実的です。

## 2. b.ai API を `.env` で設定する

手元の b.ai API が OpenAI 互換の Chat Completions エンドポイントを提供している場合、プロジェクト直下の `.env` に次を設定します。実際の URL とモデル名は b.ai の管理画面・ドキュメントの値に置き換えてください。

```dotenv
BAI_API_KEY=your-bai-api-key
BAI_API_URL=https://api.b.ai/v1/chat/completions
BAI_MODEL=your-model-name
```

`.env` は `.gitignore` 済みです。キーはソースコードや JSONL に書かないでください。別のファイルを使う場合は `NEEDLE_ENV_FILE` で指定できます。

```powershell
$env:NEEDLE_ENV_FILE = "C:\secrets\needle.env"
```

既存の DeepSeek 直接 API を使う場合は、次の設定も利用できます。

macOS / Linux:

```sh
export DEEPSEEK_API_KEY=sk-...
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

既定では次の OpenAI 互換エンドポイントを使います。

```text
https://api.deepseek.com/chat/completions
```

直接 DeepSeek のエンドポイントを変更する場合は `.env` の `DEEPSEEK_URL` で変更できます。認証情報の優先順位は `BAI_API_KEY` + `BAI_API_URL`、`DEEPSEEK_API_KEY` + `DEEPSEEK_URL`、`OPENROUTER_API_KEY` + `OPENROUTER_URL` の順です。b.ai のキーだけを設定した場合も、`https://api.b.ai/v1/chat/completions` が自動的に使われます。

## 3. ツールスキーマを用意する

`tools.json` は、実際のアプリケーションで `needle.tool` に渡すツールと同じ名前・引数定義にします。日本語の説明を入れると、生成データの品質が上がります。

```json
[
  {
    "name": "set_lights",
    "description": "部屋の照明の明るさを設定する",
    "parameters": {
      "type": "object",
      "properties": {
        "room": { "type": "string", "description": "部屋名" },
        "brightness": { "type": "integer", "description": "明るさ。0から100" }
      },
      "required": ["room", "brightness"]
    }
  }
]
```

## Google Colab で実行する

Colab では、次のセルを上から順に実行します。「ランタイム」→「ランタイムのタイプを変更」で GPU（T4 など）を選ぶと学習が速くなります。

### セル 1: リポジトリと依存関係

```python
!git clone https://github.com/cactus-compute/needle.git
%cd needle
!pip install -e .
```

### セル 2: b.ai のキーを Colab Secrets から設定

左側の鍵アイコン（Secrets）で `BAI_API_KEY` を登録し、ノートブックから読み込みます。キーをセルに直接書かないでください。

```python
import os
from google.colab import userdata

os.environ["BAI_API_KEY"] = userdata.get("BAI_API_KEY")
os.environ["BAI_API_URL"] = "https://api.b.ai/v1/chat/completions"
os.environ["BAI_MODEL"] = "your-model-name"  # b.ai で利用可能なモデル名
```

### セル 3: `tools.json` を作成またはアップロード

手元の `tools.json` を Colab のファイルパネルからアップロードするか、次のように作成します。

```python
import json

tools = [{
    "name": "set_lights",
    "description": "部屋の照明の明るさを設定する",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string"},
            "brightness": {"type": "integer"}
        },
        "required": ["room", "brightness"]
    }
}]
with open("tools.json", "w", encoding="utf-8") as f:
    json.dump(tools, f, ensure_ascii=False, indent=2)
```

### セル 4: 日本語データ生成・学習・書き出し

```python
!needle generate-data --tools tools.json --num-samples 500 --batch-size 10 --workers 4 --language ja --output data_ja.jsonl
!needle finetune data_ja.jsonl --epochs 10 --batch-size 16 --lora-rank 16 --lora-alpha 32 --max-len 1024 --out checkpoints/needle_ja_lora.pkl
!needle build checkpoints/needle2.pkl --lora checkpoints/needle_ja_lora.pkl --out needle_ja.cact --bits 2
```

### セル 5: ダウンロード

```python
from google.colab import files
files.download("needle_ja.cact")
```

Colab のセッション終了で生成データやチェックポイントが消えるため、必要に応じて Google Drive をマウントして保存してください。

セルをまとめたノートブックは [needle_japanese_finetune_colab.ipynb](needle_japanese_finetune_colab.ipynb) です。Colab で開く場合は、ファイルをアップロードしてから実行するか、GitHub 上のファイル URL を Colab の「ファイル」→「ノートブックをアップロード」から指定してください。

## 4. DeepSeek で日本語の学習データを生成する

まず少量で動作確認します。

```sh
needle generate-data \
  --tools tools.json \
  --num-samples 100 \
  --batch-size 10 \
  --language ja \
  --output data_ja.jsonl
```

生成された JSONL は、1 行 1 例で、概ね次の形式です。

```json
{"query":"キッチンの照明を10に暗くして","tools":[...],"answers":[{"name":"set_lights","arguments":{"room":"キッチン","brightness":10}}],"reasoning":"「キッチン」が部屋名、「10」が明るさを示す"}
```

本番前に次を確認してください。

- `query` と `reasoning` が日本語になっていること
- `answers` のツール名と JSON キーがスキーマと一致していること
- 引数がユーザー文に現れる値だけで構成されていること
- 対象外の問い合わせに `"answers": []` が含まれていること
- 日付、数値、固有名詞、敬語、表記ゆれを十分に含むこと

手書きの正解データを先に作り、そこから増やすこともできます。

```sh
needle generate-data \
  --augment seed_ja.jsonl \
  --num-samples 500 \
  --language ja \
  --output data_ja.jsonl
```

API の利用料金とレート制限に注意し、最初は `--num-samples 20` 程度でプロンプトとスキーマを検証してください。生成データは必ず人手またはルールで検査します。DeepSeek の出力を無検証で学習に投入すると、誤った引数対応を学習する可能性があります。

## 5. 日本語データで LoRA ファインチューニングする

```sh
needle finetune data_ja.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-len 1024 \
  --val-split 0.1 \
  --out checkpoints/needle_ja_lora.pkl
```

学習完了時には、アダプターに加えて次のファイルが自動保存されます。

- `checkpoints/needle_ja_lora.pkl.dataset.jsonl`: 実際に学習へ投入した JSONL のコピー
- `checkpoints/needle_ja_lora.pkl.metadata.json`: データ件数、系列長、ベースチェックポイント、epoch、LoRA 設定など

これにより、アダプターと学習条件を後から追跡できます。保存が不要な場合だけ `--no-save-data` を指定してください。

データ生成と学習を一度に行う場合は、次のようにします。

```sh
needle finetune seed_ja.jsonl \
  --generate 500 \
  --language ja \
  --epochs 10 \
  --out checkpoints/needle_ja_lora.pkl
```

数百例では 10〜30 epoch 程度が目安です。学習 loss が下がり続ける一方で validation loss が上がる場合は過学習なので、そこで止めるか、データを増やします。ツール選択より引数の正しいグラウンディングの方が多くの例を必要とします。

日本語は英語よりトークン数が増えやすいため、`seq_len` が `--max-len` に頻繁に達する場合は、例を短くするか `--max-len 1536` を検討してください。モデルのコンテキスト上限を超える例は切り詰められるため、短く具体的なユーザー発話を優先します。

## 6. `.cact` に書き出す

```sh
needle build checkpoints/needle2.pkl \
  --lora checkpoints/needle_ja_lora.pkl \
  --out needle_ja.cact \
  --bits 2
```

ベースチェックポイントが未取得なら、通常は Hugging Face から自動取得されます。書き出したモデルは通常の Needle と同じエンジンで動作します。

```python
import needle

agent = needle.Needle(weights="needle_ja.cact", tools=[set_lights])
result = agent.run("リビングの照明を50にして")
print(result["results"])
```

## 7. 評価と運用上の注意

学習に使っていない日本語の固定テストセットを作り、少なくとも次を評価します。

- 正しいツールを選べるか
- 数値、日付、単位、固有名詞を正しく抽出できるか
- 複数ツール呼び出しの順序と引数が正しいか
- 対象外の発話で誤ってツールを呼ばないか
- 丁寧語、口語、表記ゆれ、全角半角に耐えられるか

LoRA 学習後の `.cact` では confidence head は更新されないため、`confidence` は `None` になります。日本語運用では、confidence を安全判定の根拠にせず、固定テストセットとアプリケーション側の検証を併用してください。

また、Tokenizer は日本語専用ではありません。日本語の品質や長文対応が不足する場合、LoRA の rank や epoch を増やす前に、短く多様な日本語データを追加し、入力長とトークン化を確認してください。

## 参考コマンド

```sh
needle generate-data --help
needle finetune --help
needle build --help
```

英語版の全体説明は [README.md](README.md)、学習の技術詳細は [doc/finetuning.md](doc/finetuning.md) を参照してください。
