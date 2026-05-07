"""
SepsisAlert — Narrative Model Fine-Tuning (Option 5).

Fine-tunes a local LLM on high-rated clinician-approved narratives using
LoRA (Low-Rank Adaptation) so the model learns the exact tone, structure,
and clinical detail level that ICU staff rated highest.

─────────────────────────────────────────────────────────────────────
HARDWARE NOTE
─────────────────────────────────────────────────────────────────────
MacBook Air 8 GB RAM: too little for 7B LoRA even at 4-bit.
Recommended minimum: 16 GB RAM (M2 Pro) or a GPU with 8 GB VRAM.

For university use, run the fine-tuning step on:
  • Google Colab (free T4 GPU — 15 GB VRAM)        ← easiest
  • Kaggle Notebooks (free P100 GPU)
  • Any machine with an NVIDIA GPU

You can still run Step 1 (export) on your MacBook and copy the
finetune_pairs.jsonl to Colab for training.
─────────────────────────────────────────────────────────────────────

FULL PIPELINE
─────────────────────────────────────────────────────────────────────
Step 1 — Export training data (run on MacBook, needs feedback collected):
    python finetune_narrative.py --step export

Step 2 — Fine-tune with LoRA (run on GPU machine / Colab):
    python finetune_narrative.py --step train

Step 3 — Convert to GGUF for Ollama (run on GPU machine after training):
    python finetune_narrative.py --step convert

Step 4 — Load into Ollama (run on MacBook after copying the .gguf file):
    python finetune_narrative.py --step load
─────────────────────────────────────────────────────────────────────

REQUIREMENTS (for training step only — not in main requirements.txt)
─────────────────────────────────────────────────────────────────────
    pip install transformers>=4.40 peft>=0.10 trl>=0.8 bitsandbytes>=0.43
    pip install accelerate datasets sentencepiece
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

BASE_MODEL       = "mistralai/Mistral-7B-Instruct-v0.2"  # HuggingFace model id
LORA_OUTPUT_DIR  = Path("models/narrative_lora")          # LoRA adapter weights
GGUF_OUTPUT_DIR  = Path("models/narrative_gguf")          # converted GGUF
OLLAMA_MODEL_NAME = "sepsis-narrative"                    # name in Ollama

FINETUNE_DATA    = Path("data/feedback/finetune_pairs.jsonl")
MODELFILE_PATH   = Path("models/Modelfile.sepsis-narrative")


# ------------------------------------------------------------------ #
# Step 1 — Export training data                                        #
# ------------------------------------------------------------------ #

def step_export(min_rating: int = 4) -> None:
    """Export high-rated narrative pairs to JSONL format."""
    print("=" * 55)
    print("  Step 1 — Export fine-tuning data")
    print("=" * 55)

    from src.data.narrative_feedback import export_finetune_data

    try:
        out = export_finetune_data(min_rating=min_rating)
        # Count lines
        n = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"\n  ✅ {n} training pairs exported → {out}")
        print(f"\n  Minimum rating used: {min_rating}/5")
        print("\n  Next step (on a GPU machine):")
        print("    python finetune_narrative.py --step train")
    except ValueError as e:
        print(f"\n  ❌ {e}")
        print("  Collect more clinician ratings in the dashboard first.")
        sys.exit(1)


# ------------------------------------------------------------------ #
# Step 2 — LoRA fine-tuning                                            #
# ------------------------------------------------------------------ #

def step_train() -> None:
    """Fine-tune the base model on exported narrative pairs using LoRA."""
    print("=" * 55)
    print("  Step 2 — LoRA Fine-Tuning")
    print("=" * 55)

    # Late imports — only needed on training machine
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError:
        print(
            "\n  ❌ Training dependencies not installed.\n"
            "  Run: pip install transformers peft trl bitsandbytes "
            "accelerate datasets sentencepiece"
        )
        sys.exit(1)

    if not FINETUNE_DATA.exists():
        print(f"\n  ❌ Training data not found at {FINETUNE_DATA}")
        print("  Run step 1 first: python finetune_narrative.py --step export")
        sys.exit(1)

    n_pairs = sum(1 for _ in open(FINETUNE_DATA, encoding="utf-8"))
    print(f"\n  Training pairs: {n_pairs}")
    print(f"  Base model    : {BASE_MODEL}")
    print(f"  Output        : {LORA_OUTPUT_DIR}")

    if n_pairs < 10:
        print(
            f"\n  ⚠️  Only {n_pairs} training pairs — fine-tuning on fewer than 10 "
            "examples risks overfitting.\n"
            "  Collect more high-quality feedback before training."
        )

    # ── Load dataset ─────────────────────────────────────────────
    dataset = load_dataset("json", data_files=str(FINETUNE_DATA), split="train")

    def _format_prompt(example: dict) -> dict:
        """Convert Alpaca format to Mistral instruction format."""
        text = (
            f"<s>[INST] {example['instruction']}\n\n"
            f"{example['input']} [/INST] "
            f"{example['output']} </s>"
        )
        return {"text": text}

    dataset = dataset.map(_format_prompt)

    # ── Load base model in 4-bit (QLoRA) ─────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n  Loading base model (4-bit quantised)…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA config ───────────────────────────────────────────────
    # Only 0.1% of parameters are trained — memory-efficient
    lora_config = LoraConfig(
        r=16,                          # rank — higher = more capacity
        lora_alpha=32,                 # scaling factor
        target_modules=["q_proj", "v_proj"],  # which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",              # disable wandb / mlflow
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\n  Training…")
    trainer.train()
    trainer.save_model(str(LORA_OUTPUT_DIR))
    print(f"\n  ✅ LoRA adapter saved → {LORA_OUTPUT_DIR}")
    print("\n  Next step:")
    print("    python finetune_narrative.py --step convert")


# ------------------------------------------------------------------ #
# Step 3 — Convert to GGUF                                             #
# ------------------------------------------------------------------ #

def step_convert() -> None:
    """
    Merge LoRA adapter into the base model and convert to GGUF for Ollama.

    Requires llama.cpp to be installed and on PATH.
    Install: https://github.com/ggerganov/llama.cpp
    """
    print("=" * 55)
    print("  Step 3 — Convert to GGUF (for Ollama)")
    print("=" * 55)

    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("\n  ❌ peft / transformers not installed.")
        sys.exit(1)

    merged_dir = LORA_OUTPUT_DIR / "merged"
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Merging LoRA adapter into base model…")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(LORA_OUTPUT_DIR), device_map="cpu"
    )
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_OUTPUT_DIR))
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  Merged model saved → {merged_dir}")

    # Convert to GGUF using llama.cpp
    gguf_file = GGUF_OUTPUT_DIR / "sepsis-narrative-q4_k_m.gguf"
    print(f"\n  Converting to GGUF → {gguf_file}")
    print("  (requires llama.cpp convert script on PATH)\n")

    try:
        subprocess.run(
            [
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(merged_dir),
                "--outfile", str(gguf_file),
                "--outtype", "q4_k_m",
            ],
            check=True,
        )
        print(f"\n  ✅ GGUF file created → {gguf_file}")
        _write_modelfile(gguf_file)
        print("\n  Next step:")
        print("    python finetune_narrative.py --step load")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "\n  ❌ llama.cpp convert script not found.\n"
            "  Install llama.cpp and ensure convert_hf_to_gguf.py is available.\n"
            "  https://github.com/ggerganov/llama.cpp\n\n"
            f"  GGUF output path would be: {gguf_file}\n"
            f"  Modelfile template written to: {MODELFILE_PATH}"
        )
        _write_modelfile(gguf_file)


def _write_modelfile(gguf_path: Path) -> None:
    """Write the Ollama Modelfile for the fine-tuned model."""
    MODELFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Ollama Modelfile for SepsisAlert fine-tuned narrative model
# Generated by finetune_narrative.py

FROM {gguf_path.resolve()}

SYSTEM \"\"\"
You are a clinical decision support assistant specialising in ICU sepsis
monitoring. You generate SBAR-structured narrative alerts for bedside
nursing staff based on SHAP feature importance values from a validated
sepsis risk model. Be factual, specific, and concise. Never make
definitive diagnostic claims or prescribe treatment.
\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_predict 300
"""
    MODELFILE_PATH.write_text(content, encoding="utf-8")
    print(f"  Modelfile written → {MODELFILE_PATH}")


# ------------------------------------------------------------------ #
# Step 4 — Load into Ollama                                            #
# ------------------------------------------------------------------ #

def step_load() -> None:
    """Register the fine-tuned GGUF model with Ollama."""
    print("=" * 55)
    print("  Step 4 — Load into Ollama")
    print("=" * 55)

    if not MODELFILE_PATH.exists():
        print(f"\n  ❌ Modelfile not found at {MODELFILE_PATH}")
        print("  Run step 3 first: python finetune_narrative.py --step convert")
        sys.exit(1)

    print(f"\n  Creating Ollama model '{OLLAMA_MODEL_NAME}'…")
    try:
        subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
            check=True,
        )
        print(f"\n  ✅ Model '{OLLAMA_MODEL_NAME}' is now available in Ollama.")
        print("\n  To use it in the dashboard:")
        print(f"    Set ollama_model: \"{OLLAMA_MODEL_NAME}\" in config.yaml")
        print("    Or select it from the model dropdown in the dashboard.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "\n  ❌ Could not run 'ollama create'.\n"
            "  Make sure Ollama is installed and running (ollama serve).\n"
            f"\n  You can also run it manually:\n"
            f"    ollama create {OLLAMA_MODEL_NAME} -f {MODELFILE_PATH}"
        )


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

STEPS = {
    "export":  step_export,
    "train":   step_train,
    "convert": step_convert,
    "load":    step_load,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the SepsisAlert narrative model on clinician feedback."
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        required=True,
        help=(
            "export  — export feedback as training data (MacBook OK)\n"
            "train   — run LoRA fine-tuning        (GPU required)\n"
            "convert — convert to GGUF for Ollama  (GPU machine)\n"
            "load    — register with Ollama         (MacBook OK)\n"
        ),
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="Minimum star rating to include in training data (default: 4)",
    )
    args = parser.parse_args()

    if args.step == "export":
        step_export(min_rating=args.min_rating)
    else:
        STEPS[args.step]()
