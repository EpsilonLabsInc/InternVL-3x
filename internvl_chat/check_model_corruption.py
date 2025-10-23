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
    
    # Sample a few weights
    print(f"\nLanguage model first layer weight sample:")
    first_param = next(model.language_model.parameters())
    print(f"  Shape: {first_param.shape}")
    print(f"  Mean: {first_param.mean().item():.6f}")
    print(f"  Std: {first_param.std().item():.6f}")
    print(f"  Min: {first_param.min().item():.6f}")
    print(f"  Max: {first_param.max().item():.6f}")
    
    # Check if weights are reasonable (not all zeros or NaN)
    if torch.isnan(first_param).any():
        print("  ❌ WARNING: NaN values detected!")
    if (first_param == 0).all():
        print("  ❌ WARNING: All zeros!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
