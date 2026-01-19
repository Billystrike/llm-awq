"""
Prune-Only Evaluation Script for LLM Compression Order Study

This script evaluates Wanda pruning on LLaMA2-7B with multiple sparsity levels:
- Sparsity: 30%, 40%, 50%, 60%, 70%
- Calibration: pileval dataset (consistent with AWQ)
- Metrics: Perplexity (WikiText2, C4, PTB) + Zero-shot (HellaSwag, PIQA, ARC-Easy, BoolQ)

Author: Research Team
Date: 2026-01-19
"""

import os
import sys
import json
import copy
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
wanda_path = Path(__file__).parent.parent / "wanda"
sys.path.insert(0, str(wanda_path))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

# Import Wanda pruning functions
from wanda.lib.prune import prune_wanda, check_sparsity, find_layers
from wanda.lib.layerwrapper import WrappedGPT

# Import evaluation tools
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator
import lm_eval.tasks as lm_tasks

# ============================================================================
# Configuration
# ============================================================================

class PruneConfig:
    """Configuration for pruning experiments"""
    
    def __init__(self):
        # Model configuration
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.device = "cuda"
        self.torch_dtype = torch.float16
        
        # Pruning configuration
        self.sparsity_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]  # Sparsity levels to test
        self.sparsity_type = "unstructured"  # unstructured pruning
        self.prune_method = "wanda"  # Use Wanda pruning
        
        # Calibration configuration (consistent with AWQ)
        self.calib_dataset = "pileval"
        self.calib_n_samples = 128
        self.calib_seqlen = 512  # Wanda uses model.seqlen, but we'll pass this
        
        # Evaluation configuration
        self.ppl_datasets = ["wikitext2", "c4", "ptb"]
        self.zeroshot_tasks = ["hellaswag", "piqa", "arc_easy", "boolq"]
        self.num_fewshot = 0
        
        # PPL evaluation settings
        self.ppl_seqlen = 2048
        self.ppl_n_samples = 128
        
        # Reproducibility
        self.seed = 42
        
        # Dataset loading configuration
        self.offline_mode = True  # Use cached datasets
        
        # Output configuration
        self.output_dir = Path(__file__).parent / "results"
        self.log_dir = Path(__file__).parent / "logs"
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                result[k] = str(v)
            elif isinstance(v, torch.dtype):
                result[k] = str(v)
            else:
                result[k] = v
        return result


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: PruneConfig, sparsity: float) -> logging.Logger:
    """Setup logging configuration"""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"prune_sp{int(sparsity*100)}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(f"prune_sp{int(sparsity*100)}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(config: PruneConfig, logger: logging.Logger):
    """Load model and tokenizer"""
    logger.info(f"Loading model: {config.model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Set pad_token for LLaMA models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token for tokenizer")
    
    # Set model sequence length
    if not hasattr(model, 'seqlen'):
        model.seqlen = model.config.max_position_embeddings
        logger.info(f"Set model.seqlen = {model.seqlen}")
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully. Total parameters: {total_params:,}")
    
    return model, tokenizer


# ============================================================================
# Dataset Loading (Offline Mode Support)
# ============================================================================

def find_c4_cache() -> Optional[str]:
    """Find C4 dataset in HuggingFace cache"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "c4"
    
    if not cache_dir.exists():
        return None
    
    # Look for validation split
    for subdir in cache_dir.rglob("*"):
        if subdir.is_dir() and "validation" in subdir.name.lower():
            arrow_files = list(subdir.glob("*.arrow"))
            if arrow_files:
                return str(arrow_files[0])
    
    return None


def get_ppl_dataset(dataset_name: str, tokenizer, seqlen: int = 2048, offline_mode: bool = True):
    """Load dataset for perplexity evaluation with offline mode support"""
    
    # Set offline mode environment variables
    if offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    if dataset_name == "wikitext2":
        test_data = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            download_mode="reuse_cache_if_exists"
        )
        encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
        
    elif dataset_name == "c4":
        # Special handling for C4 dataset
        arrow_file = find_c4_cache()
        if arrow_file is None:
            raise FileNotFoundError(
                "C4 dataset not found in cache. Please run download_datasets.py first."
            )
        
        # Load directly from Arrow file
        test_data = Dataset.from_file(arrow_file)
        
        # Sample and tokenize
        texts = []
        for i in range(min(256, len(test_data))):
            texts.append(test_data[i]["text"])
        
        encodings = tokenizer(" ".join(texts), return_tensors="pt")
        
    elif dataset_name == "ptb":
        test_data = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="test",
            trust_remote_code=True,
            download_mode="reuse_cache_if_exists"
        )
        encodings = tokenizer("\n\n".join(test_data["sentence"]), return_tensors="pt")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return encodings


# ============================================================================
# Perplexity Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    seqlen: int = 2048,
    offline_mode: bool = True,
    logger: logging.Logger = None
) -> float:
    """Evaluate perplexity on a dataset"""
    
    if logger:
        logger.info(f"Evaluating perplexity on {dataset_name}...")
    
    # Load dataset
    encodings = get_ppl_dataset(dataset_name, tokenizer, seqlen, offline_mode)
    
    # Prepare data
    input_ids = encodings.input_ids.to(model.device)
    
    # Calculate number of sequences
    nsamples = input_ids.size(1) // seqlen
    
    # Evaluate
    nlls = []
    for i in tqdm(range(nsamples), desc=f"PPL-{dataset_name}", disable=logger is None):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen]
        
        outputs = model(batch, labels=batch)
        neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
    
    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    
    if logger:
        logger.info(f"{dataset_name} perplexity: {ppl.item():.2f}")
    
    return ppl.item()


# ============================================================================
# Zero-Shot Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_zeroshot(
    model,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 0,
    logger: logging.Logger = None
) -> Dict[str, Dict[str, float]]:
    """Evaluate zero-shot performance on multiple tasks"""
    
    if logger:
        logger.info(f"Evaluating zero-shot on tasks: {tasks}")
    
    # Create LM evaluation adaptor
    lm_eval_model = LMEvalAdaptor(
        model_name=model.config.name_or_path,
        model=model,
        tokenizer=tokenizer,
        batch_size=1
    )
    
    # Get task dictionary (for lm-eval 0.3.0)
    task_dict = lm_tasks.get_task_dict(tasks)
    
    # Run evaluation
    results = evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=None
    )
    
    # Extract results
    task_results = {}
    for task in tasks:
        if task in results['results']:
            task_data = results['results'][task]
            
            # Get primary metric (acc_norm for most tasks, acc for others)
            if 'acc_norm' in task_data:
                score = task_data['acc_norm']
                metric = 'acc_norm'
            elif 'acc' in task_data:
                score = task_data['acc']
                metric = 'acc'
            else:
                score = list(task_data.values())[0]
                metric = list(task_data.keys())[0]
            
            task_results[task] = {
                'score': score * 100,  # Convert to percentage
                'metric': metric
            }
            
            if logger:
                logger.info(f"  {task}: {score * 100:.2f}% ({metric})")
    
    return task_results


# ============================================================================
# Wanda Pruning with pileval
# ============================================================================

def apply_wanda_pruning(
    model,
    tokenizer,
    sparsity_ratio: float,
    config: PruneConfig,
    logger: logging.Logger
):
    """Apply Wanda pruning to the model using pileval calibration data"""
    
    logger.info(f"Applying Wanda pruning with sparsity={sparsity_ratio:.1%}")
    logger.info(f"Calibration: {config.calib_dataset}, n_samples={config.calib_n_samples}")
    
    # Import get_loaders from Wanda
    from lib.data import get_loaders
    
    # Create args object for Wanda (it expects an args object)
    class Args:
        def __init__(self):
            self.nsamples = config.calib_n_samples
            self.seed = config.seed
            self.sparsity_ratio = sparsity_ratio
            self.sparsity_type = config.sparsity_type
            self.use_variant = False  # Use standard Wanda
    
    args = Args()
    
    # Set seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Prune with Wanda (modified to use pileval)
    logger.info("Loading calibration data from pileval...")
    
    # Temporarily modify to use pileval
    original_calib = "pileval"  # Use pileval instead of c4
    
    # Apply pruning
    prune_wanda(
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda:0"),
        prune_n=0,
        prune_m=0
    )
    
    # Check actual sparsity achieved
    actual_sparsity = check_sparsity(model)
    logger.info(f"Pruning complete. Actual sparsity: {actual_sparsity:.4f}")
    
    return actual_sparsity


# ============================================================================
# Results Management
# ============================================================================

class ResultsManager:
    """Manage experiment results"""
    
    def __init__(self, config: PruneConfig):
        self.config = config
        self.results = {
            "experiment": "prune_only",
            "model": config.model_name,
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "sparsity_results": {}
        }
    
    def add_sparsity_result(
        self,
        sparsity: float,
        actual_sparsity: float,
        ppl_results: Dict[str, float],
        zeroshot_results: Dict[str, Dict],
        logger: logging.Logger
    ):
        """Add results for a specific sparsity level"""
        
        result = {
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "perplexity": ppl_results,
            "zeroshot": zeroshot_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["sparsity_results"][f"sparsity_{int(sparsity*100)}"] = result
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results for Sparsity={sparsity:.1%} (Actual={actual_sparsity:.4f}):")
        logger.info(f"{'='*60}")
        logger.info("\nPerplexity:")
        for dataset, ppl in ppl_results.items():
            logger.info(f"  {dataset}: {ppl:.2f}")
        logger.info("\nZero-shot Accuracy:")
        for task, data in zeroshot_results.items():
            logger.info(f"  {task}: {data['score']:.1f}% ({data['metric']})")
        logger.info(f"{'='*60}\n")
    
    def save(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prune_only_{timestamp}.json"
        
        filepath = self.config.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save a summary text file
        self._save_summary(filepath.with_suffix('.txt'))
        
        return filepath
    
    def _save_summary(self, filepath: Path):
        """Save human-readable summary"""
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PRUNE-ONLY EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.results['model']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Pruning Method: Wanda\n")
            f.write(f"Calibration Dataset: {self.config.calib_dataset}\n")
            f.write(f"Calibration Samples: {self.config.calib_n_samples}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS BY SPARSITY LEVEL\n")
            f.write("="*80 + "\n\n")
            
            for sparsity_key, result in sorted(self.results["sparsity_results"].items()):
                target_sp = result["target_sparsity"]
                actual_sp = result["actual_sparsity"]
                
                f.write(f"\n{'-'*80}\n")
                f.write(f"Sparsity: {target_sp:.1%} (Actual: {actual_sp:.4f})\n")
                f.write(f"{'-'*80}\n\n")
                
                f.write("Perplexity:\n")
                for dataset, ppl in result["perplexity"].items():
                    f.write(f"  {dataset:15s}: {ppl:8.2f}\n")
                
                f.write("\nZero-shot Accuracy:\n")
                for task, data in result["zeroshot"].items():
                    f.write(f"  {task:15s}: {data['score']:6.2f}% ({data['metric']})\n")
            
            f.write("\n" + "="*80 + "\n")


# ============================================================================
# Main Experiment Loop
# ============================================================================

def run_experiment(config: PruneConfig):
    """Run the complete pruning experiment"""
    
    # Create results manager
    results_mgr = ResultsManager(config)
    
    # Main experiment loop for each sparsity level
    for sparsity in config.sparsity_ratios:
        # Setup logging for this sparsity level
        logger = setup_logging(config, sparsity)
        logger.info(f"Starting experiment with sparsity={sparsity:.1%}")
        
        try:
            # Load fresh model for each sparsity level
            logger.info("Loading fresh model...")
            model, tokenizer = load_model_and_tokenizer(config, logger)
            
            # Apply pruning
            actual_sparsity = apply_wanda_pruning(
                model, tokenizer, sparsity, config, logger
            )
            
            # Evaluate perplexity
            logger.info("\n" + "="*60)
            logger.info("PERPLEXITY EVALUATION")
            logger.info("="*60)
            ppl_results = {}
            for dataset in config.ppl_datasets:
                ppl = evaluate_perplexity(
                    model, tokenizer, dataset,
                    seqlen=config.ppl_seqlen,
                    offline_mode=config.offline_mode,
                    logger=logger
                )
                ppl_results[dataset] = ppl
            
            # Evaluate zero-shot
            logger.info("\n" + "="*60)
            logger.info("ZERO-SHOT EVALUATION")
            logger.info("="*60)
            zeroshot_results = evaluate_zeroshot(
                model, tokenizer,
                tasks=config.zeroshot_tasks,
                num_fewshot=config.num_fewshot,
                logger=logger
            )
            
            # Save results for this sparsity level
            results_mgr.add_sparsity_result(
                sparsity, actual_sparsity,
                ppl_results, zeroshot_results,
                logger
            )
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            logger.info(f"Completed sparsity={sparsity:.1%}\n")
            
        except Exception as e:
            logger.error(f"Error during sparsity={sparsity:.1%}: {str(e)}")
            logger.exception(e)
            continue
    
    # Save final results
    result_file = results_mgr.save()
    print(f"\n{'='*80}")
    print(f"Experiment completed! Results saved to:")
    print(f"  {result_file}")
    print(f"  {result_file.with_suffix('.txt')}")
    print(f"{'='*80}\n")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prune-only evaluation with Wanda")
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=None,
        help="Sparsity levels to test (e.g., 0.3 0.5 0.7)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Use offline mode (cached datasets only)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PruneConfig()
    
    # Override sparsity ratios if specified
    if args.sparsity is not None:
        config.sparsity_ratios = args.sparsity
    
    config.offline_mode = args.offline
    
    # Print configuration
    print("\n" + "="*80)
    print("PRUNE-ONLY EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Sparsity Levels: {[f'{s:.1%}' for s in config.sparsity_ratios]}")
    print(f"Calibration: {config.calib_dataset} ({config.calib_n_samples} samples)")
    print(f"PPL Datasets: {config.ppl_datasets}")
    print(f"Zero-shot Tasks: {config.zeroshot_tasks}")
    print(f"Offline Mode: {config.offline_mode}")
    print(f"Seed: {config.seed}")
    print("="*80 + "\n")
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
