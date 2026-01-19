"""
Quick test script to verify Wanda pruning setup

This script performs a quick sanity check:
1. Import all required modules
2. Load a small test to verify pruning works
3. Check pileval dataset access

Run this before the full experiment to catch any issues early.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
wanda_path = Path(__file__).parent.parent / "wanda"
sys.path.insert(0, str(wanda_path))

print("="*80)
print("WANDA PRUNE-ONLY SETUP VERIFICATION")
print("="*80)

# Test 1: Import Wanda modules
print("\n[1/5] Testing Wanda imports...")
try:
    from lib.prune import prune_wanda, check_sparsity, find_layers
    from lib.layerwrapper import WrappedGPT
    from lib.data import get_loaders
    print("✅ Wanda modules imported successfully")
except Exception as e:
    print(f"❌ Failed to import Wanda modules: {e}")
    sys.exit(1)

# Test 2: Import evaluation modules
print("\n[2/5] Testing evaluation imports...")
try:
    from transformers import AutoTokenizer
    from awq.utils.lm_eval_adaptor import LMEvalAdaptor
    from lm_eval import evaluator
    import lm_eval.tasks as lm_tasks
    print("✅ Evaluation modules imported successfully")
except Exception as e:
    print(f"❌ Failed to import evaluation modules: {e}")
    sys.exit(1)

# Test 3: Check PyTorch and CUDA
print("\n[3/5] Checking PyTorch and CUDA...")
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        print(f"✅ Current device: {torch.cuda.current_device()}")
except Exception as e:
    print(f"❌ PyTorch/CUDA check failed: {e}")
    sys.exit(1)

# Test 4: Check pileval dataset cache
print("\n[4/5] Checking pileval dataset cache...")
try:
    from datasets import load_dataset
    import os
    
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Try to load pileval
    dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation')
    print(f"✅ Pileval dataset loaded: {len(dataset)} samples")
    
except Exception as e:
    print(f"⚠️  Warning: Pileval dataset not in cache: {e}")
    print("   You may need to run download_datasets.py first")

# Test 5: Check tokenizer
print("\n[5/5] Testing tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_text = "This is a test sentence."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenizer works: '{test_text}' -> {tokens.input_ids.shape[1]} tokens")
    
except Exception as e:
    print(f"⚠️  Tokenizer test failed: {e}")
    print("   Model may not be in cache")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nIf all tests passed, you can run:")
print("  python DoesOrderMatter/prune_only.py")
print("\nTo test a single sparsity level first:")
print("  python DoesOrderMatter/prune_only.py --sparsity 0.5")
print("="*80 + "\n")
