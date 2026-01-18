"""
Baseline Evaluation Script for LLM Compression Order Study

This script evaluates LLaMA2-7B on multiple metrics:
- Perplexity: WikiText2, C4, PTB
- Zero-shot: HellaSwag, PIQA, ARC-Easy, BoolQ

Additionally, it collects activation statistics for future mechanism analysis.

Author: Research Team
Date: 2026-01-16
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator

# ============================================================================
# Configuration
# ============================================================================

class ExperimentConfig:
    """Centralized configuration for reproducibility"""
    
    def __init__(self):
        # Model configuration
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.device = "cuda"
        self.torch_dtype = torch.float16
        
        # Evaluation configuration
        self.ppl_datasets = ["wikitext2", "c4", "ptb"]
        self.zeroshot_tasks = ["hellaswag", "piqa", "arc_easy", "boolq"]
        self.num_fewshot = 0
        
        # PPL evaluation settings
        self.ppl_seqlen = 2048
        self.ppl_n_samples = 128  # Number of samples for PPL evaluation
        
        # Calibration data (for future use in pruning/quantization)
        self.calib_dataset = "pileval"
        self.calib_n_samples = 128
        self.calib_seqlen = 512
        
        # Activation collection settings (for mechanism analysis)
        self.collect_activations = True
        self.activation_layers = [10, 15, 20, 25, 30]  # Representative layers
        self.activation_n_samples = 32  # Fewer samples for memory efficiency
        
        # Reproducibility
        self.seed = 42
        
        # Dataset loading configuration
        self.offline_mode = False  # Set to True to use only cached datasets
        self.download_max_retries = 3
        self.download_retry_delay = 2  # seconds
        
        # Output configuration
        self.output_dir = Path(__file__).parent / "results"
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.log_dir = Path(__file__).parent / "logs"
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                result[k] = str(v)
            elif isinstance(v, torch.dtype):
                result[k] = str(v)
            elif isinstance(v, list):
                result[k] = v
            else:
                result[k] = v
        return result
    
    def save(self, filepath: Path):
        """Save configuration to JSON"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: ExperimentConfig) -> logging.Logger:
    """Setup logging with both file and console handlers"""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"baseline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(
    config: ExperimentConfig,
    logger: logging.Logger
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with proper configuration"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Set pad_token for models that don't have one (like LLaMA)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    
    # Log model info
    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6
    logger.info(f"Model loaded: {param_count_m:.2f}M parameters")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


# ============================================================================
# Perplexity Evaluation
# ============================================================================

def find_c4_cache():
    """Find C4 dataset in cache directory"""
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    # Look for allenai___c4 directories
    c4_dirs = list(cache_dir.glob("allenai___c4/*"))
    
    if not c4_dirs:
        return None
    
    # Return the first valid cache directory
    for c4_dir in c4_dirs:
        # Check if it has validation split
        validation_paths = [
            c4_dir / "validation",
            c4_dir / "en" / "validation",
        ]
        for val_path in validation_paths:
            if val_path.exists():
                return c4_dir
    
    return c4_dirs[0] if c4_dirs else None


def get_ppl_dataset(dataset_name: str, tokenizer, seqlen: int, n_samples: int, offline_mode: bool = False):
    """Load and prepare dataset for PPL evaluation"""
    import time
    import os
    
    # Set offline mode environment variable to prevent any network access
    if offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Set download mode
    download_mode = "force_redownload" if not offline_mode else None
    
    max_retries = 1 if offline_mode else 3
    
    for attempt in range(max_retries):
        try:
            if dataset_name == "wikitext2":
                test_data = load_dataset(
                    "wikitext", 
                    "wikitext-2-raw-v1", 
                    split="test",
                    download_mode="reuse_cache_if_exists"
                )
                test_enc = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
                
            elif dataset_name == "c4":
                # Use completely offline mode for C4 to avoid network issues
                from datasets import load_from_disk, Dataset
                import glob
                
                c4_cache = find_c4_cache()
                
                if not c4_cache:
                    raise RuntimeError(
                        f"C4 dataset not found in cache. "
                        f"Please run 'python download_datasets.py' first."
                    )
                
                logging.info(f"Loading C4 from cache: {c4_cache}")
                
                # Force complete offline mode
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                
                # Try multiple loading strategies
                val_data = None
                errors = []
                
                # Strategy 1: load_from_disk on specific subfolders
                for subpath in ["validation", "en-validation", "default-c7bc8b0aefc5e48f"]:
                    if val_data is not None:
                        break
                    val_path = c4_cache / subpath
                    if val_path.exists():
                        try:
                            val_data = load_from_disk(str(val_path))
                            logging.info(f"Successfully loaded from {val_path}")
                            break
                        except Exception as e:
                            errors.append(f"load_from_disk({subpath}): {e}")
                
                # Strategy 2: Find and load Arrow files directly
                if val_data is None:
                    try:
                        # Look for validation arrow files
                        arrow_patterns = [
                            str(c4_cache / "**" / "*validation*.arrow"),
                            str(c4_cache / "**" / "*.arrow"),
                        ]
                        
                        arrow_files = []
                        for pattern in arrow_patterns:
                            arrow_files.extend(glob.glob(pattern, recursive=True))
                        
                        if arrow_files:
                            # Filter for validation files
                            validation_files = [f for f in arrow_files if 'validation' in f.lower()]
                            if not validation_files:
                                validation_files = arrow_files  # Use any available
                            
                            logging.info(f"Found Arrow files: {validation_files[:3]}")
                            
                            # Load directly from Arrow file using Dataset.from_file()
                            arrow_file = validation_files[0]
                            val_data = Dataset.from_file(arrow_file)
                            logging.info(f"Successfully loaded from Arrow file: {arrow_file}")
                        else:
                            errors.append("No Arrow files found in cache")
                    except Exception as e:
                        errors.append(f"Arrow file loading: {e}")
                
                # Strategy 3: Try loading the entire cache directory
                if val_data is None:
                    try:
                        val_data = load_from_disk(str(c4_cache))
                        logging.info(f"Successfully loaded entire cache directory")
                    except Exception as e:
                        errors.append(f"load entire cache: {e}")
                
                if val_data is None:
                    raise RuntimeError(
                        f"Failed to load C4 from cache at {c4_cache}. "
                        f"Tried multiple strategies. Errors:\\n" + "\\n".join(errors)
                    )
                
                # Ensure we have a validation split
                if hasattr(val_data, 'keys') and 'validation' in val_data:
                    val_data = val_data['validation']
                
                val_enc = tokenizer(" ".join(val_data[:1100]["text"]), return_tensors="pt")
                test_enc = val_enc
                
            elif dataset_name == "ptb":
                test_data = load_dataset(
                    "ptb_text_only", 
                    "penn_treebank", 
                    split="test", 
                    trust_remote_code=True,
                    download_mode="reuse_cache_if_exists"
                )
                test_enc = tokenizer(" ".join(test_data["sentence"]), return_tensors="pt")
                
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            return test_enc
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Failed to load {dataset_name} (attempt {attempt + 1}/{max_retries}): {e}")
                logging.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to load {dataset_name} after {max_retries} attempts: {e}")


@torch.no_grad()
def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    config: ExperimentConfig,
    logger: logging.Logger
) -> Dict:
    """Evaluate perplexity on a specific dataset"""
    logger.info(f"Evaluating PPL on {dataset_name}...")
    
    test_enc = get_ppl_dataset(dataset_name, tokenizer, config.ppl_seqlen, config.ppl_n_samples, config.offline_mode)
    test_ids = test_enc.input_ids
    
    # Check if we have enough data
    if test_ids.numel() < config.ppl_seqlen:
        logger.warning(f"Dataset {dataset_name} has insufficient tokens ({test_ids.numel()} < {config.ppl_seqlen})")
        raise ValueError(f"Insufficient data for {dataset_name}")
    
    nsamples = test_ids.numel() // config.ppl_seqlen
    
    # Limit samples if specified
    if config.ppl_n_samples > 0:
        nsamples = min(nsamples, config.ppl_n_samples)
    
    if nsamples == 0:
        raise ValueError(f"No samples available for {dataset_name}")
    
    nlls = []
    logger.info(f"Number of samples: {nsamples}")
    
    for i in tqdm(range(nsamples), desc=f"PPL-{dataset_name}"):
        batch = test_ids[:, (i * config.ppl_seqlen):((i + 1) * config.ppl_seqlen)].to(model.device)
        
        outputs = model(batch, labels=batch)
        neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    logger.info(f"{dataset_name} PPL: {ppl:.4f}")
    
    return {
        "dataset": dataset_name,
        "ppl": ppl,
        "nsamples": nsamples,
        "seqlen": config.ppl_seqlen
    }


# ============================================================================
# Zero-shot Evaluation
# ============================================================================

def evaluate_zeroshot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tasks: List[str],
    num_fewshot: int,
    logger: logging.Logger
) -> Dict:
    """Evaluate zero-shot performance using lm-eval-harness"""
    logger.info(f"Evaluating zero-shot tasks: {tasks}")
    
    # Create LM evaluator adaptor
    lm = LMEvalAdaptor(
        model_name=model.config.model_type,
        model=model,
        tokenizer=tokenizer,
        batch_size=1
    )
    
    # Run evaluation - use evaluator.evaluate for lm-eval 0.3.0
    from lm_eval import tasks as lm_tasks
    
    task_dict = lm_tasks.get_task_dict(tasks)
    
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=None,
        bootstrap_iters=0,  # Disable bootstrap for speed
    )
    
    # Extract and format results
    formatted_results = {}
    for task in tasks:
        if task in results["results"]:
            task_result = results["results"][task]
            # Extract accuracy metric (different tasks may have different metric names)
            if "acc" in task_result:
                acc = task_result["acc"]
            elif "acc_norm" in task_result:
                acc = task_result["acc_norm"]
            else:
                acc = list(task_result.values())[0]  # Take first metric
            
            formatted_results[task] = {
                "accuracy": acc,
                "all_metrics": task_result
            }
            logger.info(f"{task}: {acc:.4f}")
    
    return formatted_results


# ============================================================================
# Activation Collection (for Future Mechanism Analysis)
# ============================================================================

class ActivationCollector:
    """Collect activations from specific layers for mechanism analysis"""
    
    def __init__(self, model, layer_indices: List[int], device: str = "cuda"):
        self.model = model
        self.layer_indices = layer_indices
        self.device = device
        self.activations = {idx: [] for idx in layer_indices}
        self.hooks = []
        
    def _make_hook(self, layer_idx: int):
        """Create hook function for a specific layer"""
        def hook_fn(module, input, output):
            # Store activation statistics (not full tensors to save memory)
            if isinstance(output, tuple):
                output = output[0]  # For layers that return tuples
            
            act = output.detach().to(torch.float32)

            # Collect statistics instead of full activations
            act_mean = act.mean()
            act_std = act.std()

            # Clamp std to avoid division by zero in outlier computation
            act_std = torch.clamp(act_std, min=1e-6)

            stats = {
                "mean": act_mean.item(),
                "std": act_std.item(),
                "max": act.max().item(),
                "min": act.min().item(),
                "q99": act.quantile(0.99).item(),
                "q95": act.quantile(0.95).item(),
                "q90": act.quantile(0.90).item(),
                # Outlier detection (>6 sigma)
                "outlier_ratio_6sigma": (act.abs() > (act_mean.abs() + 6 * act_std)).float().mean().item(),
                "outlier_ratio_4sigma": (act.abs() > (act_mean.abs() + 4 * act_std)).float().mean().item(),
            }
            
            self.activations[layer_idx].append(stats)
        
        return hook_fn
    
    def register_hooks(self):
        """Register forward hooks to collect activations"""
        model_layers = self.model.model.layers  # For LLaMA architecture
        
        for layer_idx in self.layer_indices:
            # Hook the MLP output (key for AWQ analysis)
            layer = model_layers[layer_idx]
            hook = layer.mlp.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def aggregate_statistics(self) -> Dict:
        """Aggregate collected statistics"""
        aggregated = {}
        
        for layer_idx, stats_list in self.activations.items():
            if not stats_list:
                continue
            
            # Average statistics across all samples
            aggregated[f"layer_{layer_idx}"] = {
                key: np.mean([s[key] for s in stats_list])
                for key in stats_list[0].keys()
            }
        
        return aggregated


@torch.no_grad()
def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    logger: logging.Logger
) -> Dict:
    """Collect activation statistics for mechanism analysis"""
    
    if not config.collect_activations:
        logger.info("Activation collection disabled")
        return {}
    
    logger.info("Collecting activation statistics...")
    
    # Setup activation collector
    collector = ActivationCollector(
        model=model,
        layer_indices=config.activation_layers,
        device=config.device
    )
    collector.register_hooks()
    
    # Load calibration data
    from awq.utils.calib_data import get_calib_dataset
    calib_data = get_calib_dataset(
        data=config.calib_dataset,
        tokenizer=tokenizer,
        n_samples=config.activation_n_samples,
        block_size=config.calib_seqlen
    )
    
    # Forward pass through calibration data
    for i, batch in enumerate(tqdm(calib_data, desc="Collecting activations")):
        try:
            batch = batch.to(model.device)
            _ = model(batch)
        except Exception as e:
            logger.warning(f"Error processing batch {i}: {e}")
            continue
        
        if i >= config.activation_n_samples:
            break
    
    # Aggregate statistics
    activation_stats = collector.aggregate_statistics()
    collector.remove_hooks()
    
    logger.info(f"Collected activation statistics from {len(config.activation_layers)} layers")
    
    return activation_stats


# ============================================================================
# Results Management
# ============================================================================

class ResultsManager:
    """Manage experiment results and ensure reproducibility"""
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "model": config.model_name,
                "config": config.to_dict()
            },
            "perplexity": {},
            "zeroshot": {},
            "activations": {}
        }
        
        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def add_ppl_result(self, result: Dict):
        """Add perplexity result"""
        dataset = result["dataset"]
        self.results["perplexity"][dataset] = result
    
    def add_zeroshot_results(self, results: Dict):
        """Add zero-shot results"""
        self.results["zeroshot"] = results
    
    def add_activation_stats(self, stats: Dict):
        """Add activation statistics"""
        self.results["activations"] = stats
    
    def save(self, filename: str = "baseline_fp16.json"):
        """Save results to JSON file"""
        output_path = self.config.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Also save a summary
        self._save_summary()
    
    def _save_summary(self):
        """Save human-readable summary"""
        summary_path = self.config.output_dir / "baseline_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BASELINE EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Timestamp: {self.results['experiment_info']['timestamp']}\n\n")
            
            # Perplexity results
            f.write("-" * 80 + "\n")
            f.write("PERPLEXITY RESULTS\n")
            f.write("-" * 80 + "\n")
            for dataset, result in self.results["perplexity"].items():
                f.write(f"{dataset:15s}: {result['ppl']:8.4f}\n")
            
            # Zero-shot results
            f.write("\n" + "-" * 80 + "\n")
            f.write("ZERO-SHOT RESULTS\n")
            f.write("-" * 80 + "\n")
            for task, result in self.results["zeroshot"].items():
                f.write(f"{task:15s}: {result['accuracy']:8.4f}\n")
            
            # Activation statistics (summary)
            if self.results["activations"]:
                f.write("\n" + "-" * 80 + "\n")
                f.write("ACTIVATION STATISTICS (Outlier Ratios)\n")
                f.write("-" * 80 + "\n")
                for layer, stats in self.results["activations"].items():
                    f.write(f"{layer:15s}: 6σ={stats['outlier_ratio_6sigma']:.6f}, "
                           f"4σ={stats['outlier_ratio_4sigma']:.6f}\n")
        
        self.logger.info(f"Summary saved to {summary_path}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main(args):
    """Main evaluation pipeline"""
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Override config with command line arguments if provided
    if args.model:
        config.model_name = args.model
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if not args.collect_activations:
        config.collect_activations = False
    if args.offline:
        config.offline_mode = True
    
    # Setup logging
    logger = setup_logging(config)
    
    if config.offline_mode:
        logger.info("Running in OFFLINE mode - using only cached datasets")
    logger.info("=" * 80)
    logger.info("BASELINE EVALUATION START")
    logger.info("=" * 80)
    
    # Save configuration
    config.save(config.output_dir / "config.json")
    
    # Initialize results manager
    results_mgr = ResultsManager(config, logger)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config, logger)
        
        # =====================================================================
        # Phase 1: Perplexity Evaluation
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: PERPLEXITY EVALUATION")
        logger.info("=" * 80)
        
        for dataset in config.ppl_datasets:
            ppl_result = evaluate_perplexity(model, tokenizer, dataset, config, logger)
            results_mgr.add_ppl_result(ppl_result)
        
        # =====================================================================
        # Phase 2: Zero-shot Evaluation
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: ZERO-SHOT EVALUATION")
        logger.info("=" * 80)
        
        zeroshot_results = evaluate_zeroshot(
            model, tokenizer, config.zeroshot_tasks, config.num_fewshot, logger
        )
        results_mgr.add_zeroshot_results(zeroshot_results)
        
        # =====================================================================
        # Phase 3: Activation Collection (for future analysis)
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: ACTIVATION STATISTICS COLLECTION")
        logger.info("=" * 80)
        
        activation_stats = collect_activations(model, tokenizer, config, logger)
        results_mgr.add_activation_stats(activation_stats)
        
        # =====================================================================
        # Save Results
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        results_mgr.save()
        
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE EVALUATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for LLM compression order study"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (default: meta-llama/Llama-2-7b-hf)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-collect-activations",
        dest="collect_activations",
        action="store_false",
        help="Disable activation collection"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached datasets (no internet required)"
    )
    
    args = parser.parse_args()
    
    main(args)
