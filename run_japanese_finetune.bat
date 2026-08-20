@echo off
setlocal

if not exist ".venv\Scripts\needle.exe" (
  echo .venv is not ready. Run setup.bat first.
  exit /b 1
)
if not exist "tools.json" (
  echo tools.json was not found.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
needle generate-data --tools tools.json --num-samples 500 --batch-size 5 --workers 4 --language ja --model deepseek-v4-flash --output data_ja.jsonl
if errorlevel 1 exit /b 1
needle finetune data_ja.jsonl --epochs 10 --batch-size 16 --lora-rank 16 --lora-alpha 32 --max-len 1024 --out checkpoints\needle_ja_lora.pkl
if errorlevel 1 exit /b 1
needle build checkpoints\needle2.pkl --lora checkpoints\needle_ja_lora.pkl --out needle_ja.cact --bits 2
if errorlevel 1 exit /b 1

echo Done: needle_ja.cact
