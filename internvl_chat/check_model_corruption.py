import torch
from internvl.model.internvl_chat import InternVLChatModel

# Check a merged checkpoint
model_path = "/home/ruian/projects/InternVL-3x/internvl_chat/training/internvl_chat_v3_mpo/lora_merged/test1"
#model_path = "/home/ruian/projects/InternVL-3x/internvl_chat/training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_064500_1e-6"
#model_path = "/home/ruian/vlm_ckpt_v2.0/label/internvl3_chimera_20251009_004033_1e-5_consolidated_labels-1009-38B-8B/checkpoint-23688/"
model_path = "/home/ruian/vlm_ckpt_v2.0/label/internvl3_chimera_20251009_004033_1e-5_consolidated_labels-1009-38B-8B/"

#model_path = "/home/ruian/projects/InternVL-3x/internvl_chat/training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_192201_1e-6"

#model_path = "/home/ruian/projects/InternVL-3x/internvl_chat/"
#model_path += "training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_200157_4e-8"
#model_path += "training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_201712_4e-8/"
#model_path += "training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_202536_1e-10"
#model_path += "training/internvl_chat_v3_mpo/Internvl3.0_chimera-38b-8b_mpo_20251023_204404_1e-10/checkpoint-1/"


print(model_path)

def analyze_module_weights(module, module_name, max_params=20):
    """Analyze weights in a module, showing multiple parameters"""
    print(f"\n{'='*80}")
    print(f"📊 {module_name} Weight Analysis")
    print(f"{'='*80}")
    
    param_list = []
    for name, param in module.named_parameters():
        if param.numel() > 0:  # Skip empty tensors
            param_list.append((name, param))
    
    print(f"Total parameters: {len(param_list)}")
    print(f"Showing first {min(max_params, len(param_list))} parameters:\n")
    
    for idx, (name, param) in enumerate(param_list[:max_params]):
        print(f"[{idx+1}] {name}")
        print(f"    Shape: {param.shape}, Numel: {param.numel()}")
        
        # Safe statistics computation
        if param.numel() > 0:
            mean_val = param.mean().item()
            std_val = param.std().item()
            min_val = param.min().item()
            max_val = param.max().item()
            num_zeros = (param == 0).sum().item()
            zero_pct = (num_zeros / param.numel()) * 100
            
            print(f"    Mean: {mean_val:>12.8f}  Std: {std_val:>12.8f}")
            print(f"    Min:  {min_val:>12.8f}  Max: {max_val:>12.8f}")
            print(f"    Zeros: {num_zeros}/{param.numel()} ({zero_pct:.2f}%)")
            
            # Health checks
            issues = []
            if torch.isnan(param).any():
                issues.append("⚠️  NaN values detected")
            if torch.isinf(param).any():
                issues.append("⚠️  Inf values detected")
            if zero_pct == 100:
                issues.append("❌ ALL ZEROS")
            if std_val == 0:
                issues.append("⚠️  Zero std (constant values)")
            
            if issues:
                for issue in issues:
                    print(f"    {issue}")
        print()

try:
    model = InternVLChatModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    print("\n✓ Model loaded successfully")
    print(f"\nModel config:")
    print(f"  use_llm_lora: {model.config.use_llm_lora}")
    print(f"  use_backbone_lora: {model.config.use_backbone_lora}")
    
    # Check language model
    print(f"\nLanguage model type: {type(model.language_model)}")
    print(f"Language model has 'merge_and_unload': {hasattr(model.language_model, 'merge_and_unload')}")
    
    # Check vision model
    print(f"\nVision model type: {type(model.vision_model)}")
    print(f"Vision model has 'merge_and_unload': {hasattr(model.vision_model, 'merge_and_unload')}")
    
    # Analyze language model weights
    analyze_module_weights(model.language_model, "Language Model (LLM)", max_params=15)
    
    # Analyze vision model weights
    analyze_module_weights(model.vision_model, "Vision Model (ViT)", max_params=10)
    
    # Analyze MLP weights if available
    if hasattr(model, 'mlp1'):
        analyze_module_weights(model.mlp1, "MLP Projector", max_params=5)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("📈 Overall Model Summary")
    print(f"{'='*80}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_zeros = sum((p == 0).sum().item() for p in model.parameters() if p.numel() > 0)
    total_nans = sum(torch.isnan(p).sum().item() for p in model.parameters() if p.numel() > 0)
    total_infs = sum(torch.isinf(p).sum().item() for p in model.parameters() if p.numel() > 0)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Total zeros: {total_zeros:,} ({100*total_zeros/total_params:.2f}%)")
    print(f"Total NaNs: {total_nans:,}")
    print(f"Total Infs: {total_infs:,}")
    
    if total_nans > 0 or total_infs > 0:
        print(f"\n❌ CRITICAL: Model contains {total_nans} NaN and {total_infs} Inf values!")
    elif 100*total_zeros/total_params > 50:
        print(f"\n⚠️  WARNING: Model has >50% zero values - may indicate undertrained weights")
    else:
        print(f"\n✓ Model weights appear healthy")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
