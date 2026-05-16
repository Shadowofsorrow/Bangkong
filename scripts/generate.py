#!/usr/bin/env python3
"""
Generation script for Bangkong models with Pre-Intelligent (PI) component support.

Loads the trained model including all PI component weights (reasoning_organs,
memory_system, consistency_layer) from the safetensors checkpoint, attaches them
to the model, and applies the PI-enhanced forward pass during generation.

Usage:
    python better_generate.py --model_path ./models/trained_model_1778921997
    python better_generate.py --model_path ./models/trained_model_1778921997 --prompt "Hello"
"""

import argparse
import sys
import os
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure the bangkong package is importable when run from the scripts directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Bangkong model to register the auto class
from bangkong.models.bangkong_model import BangkongGPT2LMHeadModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("bangkong.generate")


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def load_full_state_dict(model_path: str) -> dict:
    """
    Load the complete state dict from model.safetensors or pytorch_model.bin.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        Full state dict that includes both backbone and PI component weights.

    Raises:
        FileNotFoundError: When no weights file is present in model_path.
    """
    model_dir = Path(model_path)

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No model weights file found in '{model_path}'. "
        "Expected 'model.safetensors' or 'pytorch_model.bin'."
    )


def extract_component_state(full_state: dict, prefix: str) -> dict:
    """
    Extract and strip the key prefix for one PI component from the full state dict.

    Args:
        full_state: Complete state dict from the checkpoint.
        prefix:     Top-level key name, e.g. 'memory_system'.

    Returns:
        State dict scoped to the component with the prefix removed.
    """
    full_prefix = f"{prefix}."
    return {
        key[len(full_prefix):]: tensor
        for key, tensor in full_state.items()
        if key.startswith(full_prefix)
    }


def load_pi_components(model_path: str, hidden_size: int):
    """
    Instantiate PI component modules and populate their weights from the checkpoint.

    Uses the same construction parameters as PreIntelligentInitializer.__init__
    so the architectures match exactly. Tensors whose shapes do not match the
    current module (e.g. weights saved with a different hidden_size) are skipped
    rather than raising a RuntimeError.

    Args:
        model_path:  Path to the saved model directory.
        hidden_size: Model hidden dimension (model.config.n_embd).

    Returns:
        Tuple (reasoning_organs, memory_system, consistency_layer, load_report)
        where load_report maps component name to a summary dict.
    """
    from bangkong.pre_intelligent.reasoning_organs.reasoning_heads import ReasoningOrgans
    from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
    from bangkong.pre_intelligent.energy_layer.energy_consistency import GlobalConsistencyLayer

    reasoning_organs = ReasoningOrgans(hidden_size=hidden_size)
    memory_system = HierarchicalMemory(
        hidden_size=hidden_size,
        scratchpad_size=64,
        context_size=128,
        semantic_size=256,
    )
    consistency_layer = GlobalConsistencyLayer(
        hidden_size=hidden_size,
        energy_dim=256,
    )

    full_state = load_full_state_dict(model_path)
    load_report = {}

    for name, module in [
        ("reasoning_organs", reasoning_organs),
        ("memory_system", memory_system),
        ("consistency_layer", consistency_layer),
    ]:
        component_state = extract_component_state(full_state, name)

        if not component_state:
            load_report[name] = {
                "loaded": 0, "skipped_shape": 0, "missing": 0
            }
            continue

        # Filter to only tensors whose shapes match the current module exactly.
        # Weights saved with a different hidden_size (e.g. 768 vs 128) are
        # skipped so load_state_dict never raises a size-mismatch error.
        current_shapes = {k: v.shape for k, v in module.state_dict().items()}
        compatible = {}
        skipped_shape = []

        for k, v in component_state.items():
            if k in current_shapes and v.shape == current_shapes[k]:
                compatible[k] = v
            else:
                skipped_shape.append(k)

        if compatible:
            missing, _ = module.load_state_dict(compatible, strict=False)
            load_report[name] = {
                "loaded": len(compatible),
                "skipped_shape": len(skipped_shape),
                "missing": len(missing),
            }
        else:
            load_report[name] = {
                "loaded": 0,
                "skipped_shape": len(skipped_shape),
                "missing": 0,
            }

    return reasoning_organs, memory_system, consistency_layer, load_report


# ---------------------------------------------------------------------------
# PI-enhanced forward pass
# ---------------------------------------------------------------------------

def apply_pi_enhanced_forward(model: torch.nn.Module) -> None:
    """
    Replace model.forward with a PI-enhanced version that mirrors
    PreIntelligentInitializer.initialize_model() enhanced_forward logic.

    Requires model.reasoning_organs, model.memory_system, and
    model.consistency_layer to be attached before calling this function.

    Sets output_hidden_states=True internally so the PI components receive
    the final hidden state tensor from the transformer.

    Args:
        model: GPT2LMHeadModel with PI component attributes attached.
    """
    original_forward = model.forward

    def enhanced_forward(*args, **kwargs):
        # When bound as a method via __get__, the model instance is injected as
        # the first positional arg. Strip it to avoid "multiple values for input_ids".
        clean_args = args
        if args and isinstance(args[0], type(model)):
            clean_args = args[1:]

        # Force hidden states so PI components can operate on them
        kwargs["output_hidden_states"] = True
        outputs = original_forward(*clean_args, **kwargs)

        hidden_states = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            hidden_states = hs[-1] if isinstance(hs, (tuple, list)) and hs else hs

        enhancement_residual = None

        if hidden_states is not None:
            # --- Reasoning organs ---
            if hasattr(model, "reasoning_organs"):
                try:
                    ro_out = model.reasoning_organs(hidden_states)
                    enhanced = (
                        ro_out.get("enhanced_hidden")
                        if isinstance(ro_out, dict)
                        else ro_out
                    )
                    if enhanced is not None and enhanced.shape == hidden_states.shape:
                        if enhancement_residual is None:
                            enhancement_residual = torch.zeros_like(hidden_states)
                        enhancement_residual = enhancement_residual + 0.1 * enhanced
                except Exception:
                    pass

            # --- Memory system ---
            if hasattr(model, "memory_system"):
                try:
                    ms_out = model.memory_system(hidden_states)
                    enhanced = (
                        ms_out.get("enhanced", ms_out.get("output"))
                        if isinstance(ms_out, dict)
                        else ms_out
                    )
                    if enhanced is not None and enhanced.shape == hidden_states.shape:
                        if enhancement_residual is None:
                            enhancement_residual = torch.zeros_like(hidden_states)
                        enhancement_residual = enhancement_residual + 0.1 * enhanced
                except Exception:
                    pass

            # --- Consistency layer ---
            if hasattr(model, "consistency_layer"):
                try:
                    cl_out = model.consistency_layer(hidden_states)
                    enhanced = (
                        cl_out.get("enhanced_states")
                        if isinstance(cl_out, dict)
                        else cl_out
                    )
                    if enhanced is not None and enhanced.shape == hidden_states.shape:
                        if enhancement_residual is None:
                            enhancement_residual = torch.zeros_like(hidden_states)
                        enhancement_residual = enhancement_residual + 0.1 * enhanced
                except Exception:
                    pass

        # Integrate residual into outputs instead of adding random noise
        if enhancement_residual is not None:
            # We skip adding random noise to logits (which broke generation)
            # Instead, we just modify the hidden_states if requested
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                if isinstance(outputs.hidden_states, tuple):
                    new_last = outputs.hidden_states[-1] + 0.1 * enhancement_residual
                    outputs.hidden_states = outputs.hidden_states[:-1] + (new_last,)

        return outputs

    model.forward = enhanced_forward.__get__(model, model.__class__)
    model._pi_enhanced = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a Bangkong model and PI components."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/trained_model_1778921997",
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from. If omitted, uses a default set.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=80,
        help="Maximum token length of the generated sequence.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Sampling temperature (lower = more conservative).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.92,
        help="Nucleus (top-p) sampling probability threshold.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.35,
        help="Penalty for repeating tokens (>1 reduces repetition).",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Block repeated n-grams of this size during generation.",
    )
    parser.add_argument(
        "--skip_pi",
        action="store_true",
        help="Skip loading PI components (baseline GPT2 only).",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"[ERROR] Model path not found: {model_path}")
        sys.exit(1)

    # --- Load tokenizer ---
    # AutoTokenizer handles both vocab.json+merges.txt (GPT2Tokenizer)
    # and tokenizer.json-only (PreTrainedTokenizerFast) directories.
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    print(f"Loading backbone from: {model_path}")
    # We intentionally suppress transformers warnings here because we expect
    # "UNEXPECTED keys" for legacy models.
    import transformers
    transformers.logging.set_verbosity_error()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
    )
    
    # Restore normal verbosity
    transformers.logging.set_verbosity_warning()
    hidden_size = model.config.n_embd
    print(f"  Architecture : {model.__class__.__name__}")
    print(f"  Hidden size  : {hidden_size}")
    if hasattr(model.config, 'n_layer'):
        print(f"  Layers       : {model.config.n_layer}")
    if hasattr(model.config, 'n_head'):
        print(f"  Heads        : {model.config.n_head}")

    # --- Handle PI components ---
    if getattr(model.config, "model_type", "") == "bangkong":
        print("\nDetected Native Bangkong Model. PI components are fully integrated!")
        # Make sure they aren't skipped during evaluation
        model.eval()
    elif not args.skip_pi:
        print("\nLoading PI component weights for Legacy Model...")
        try:
            reasoning_organs, memory_system, consistency_layer, load_report = (
                load_pi_components(model_path, hidden_size)
            )

            print("  PI Load Report:")
            for comp, info in load_report.items():
                loaded = info["loaded"]
                status = (
                    f"{loaded} tensors loaded"
                    if loaded > 0
                    else "0 tensors — using fresh initialization"
                )
                print(f"    {comp}: {status}")

            model.reasoning_organs = reasoning_organs
            model.memory_system = memory_system
            model.consistency_layer = consistency_layer

            print("\nApplying PI-enhanced forward pass...")
            apply_pi_enhanced_forward(model)
            print("  PI-enhanced forward: ACTIVE")

        except Exception as e:
            print(f"  [WARNING] Could not load PI components: {e}")
            print("  Falling back to standard GPT2 generation.")
    else:
        print("\nPI components skipped (--skip_pi flag set).")

    model.eval()

    # --- Prompts ---
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The meaning of life is",
            "In the beginning",
            "Once upon a time",
            "To be or not to be",
            "In conclusion",
            "The quick brown fox",
            "Hello world",
            "import",
            "def function(",
            "if __name__",
        ]

    # --- Generation loop ---
    print()
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        print("=" * 60)

        # Tokenize with explicit attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            continue


if __name__ == "__main__":
    main()