
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import torch.nn.functional as F
import timm
from torchvision import transforms as T
from urllib.request import urlopen

def visualize_radio():
    torch.set_grad_enabled(False) # Globally disable gradients
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Image
    from urllib.request import urlopen
    image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    print(f"Loading image from URL: {image_url}")
    image = Image.open(urlopen(image_url)).convert('RGB')
    print(f"Original image size: {image.size}")

    # Resize for AnyUp/RADIO input. 
    # To reduce VRAM usage, use a slightly smaller size if 512 is OOM.
    # RADIO standard is often smaller.
    target_size = (448, 448) 
    print(f"Resizing image to {target_size} for model input...")
    image_resized = image.resize(target_size, Image.Resampling.BICUBIC)

    # 2. Load RADIO v4 Model
    hf_repo = "nvidia/C-RADIOv4-H" # Correct name is "nvidia/C-RADIOv4-H" or "nvidia/radio-v4" ? Check user provided: "nvidia/C-RADIOv4-H"
    print(f"Loading model from {hf_repo}...")
    try:
        # Load processor
        image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        
        # Load model using safetensors if possible
        model = AutoModel.from_pretrained(
            hf_repo, 
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Failed to load model from {hf_repo} with safetensors: {e}")
        # Fallback without safetensors flag
        print("Trying fallback without safetensors flag...")
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Check if we can fit on GPU, if not fallback to CPU
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"Could not load model to GPU: {e}. Falling back to CPU")
        device = "cpu"
        model.to(device)
    
    print("Model loaded successfully.")

    # 3. Preprocess Image
    # Pass resized image to processor, but ensure it doesn't downsample again if not needed
    # Some processors force specific size. We can pass return_tensors='pt' and handle resizing manually if needed.
    # But usually processor accepts 'size' argument.
    # For now, let pass resized image and disable processor resize to trust our resize.
    try:
        inputs = image_processor(images=image_resized, return_tensors='pt', do_resize=False)
        pixel_values = inputs.pixel_values.to(device)
    except Exception as e:
        print(f"Processor failed without resize, trying with default resize: {e}")
        inputs = image_processor(images=image, return_tensors='pt', do_resize=True)
        pixel_values = inputs.pixel_values.to(device)
        
    print(f"Pixel values shape: {pixel_values.shape}")

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device=="cuda" else "cpu"): # Use mixed precision
            summary, features = model(pixel_values)
            
    print(f"Summary shape: {summary.shape}")
    print(f"Features shape: {features.shape}")
    
    # Detach features from any model reference effectively
    features = features.clone()
    
    # Offload model to CPU to free VRAM for AnyUp and PCA
    print("Deleting model to free VRAM...", flush=True)
    try:
        # model.cpu() # Skipping explicit move to CPU to avoid RAM spike
        del model
        del image_processor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Model deleted and cache cleared.", flush=True)
    except Exception as e:
        print(f"Error during model deletion: {e}", flush=True)

    # Features shape is likely (B, N, C). we need to determine grid size HxW.
    # N is sequence length.
    B, N, C = features.shape
    
    H_img, W_img = pixel_values.shape[2], pixel_values.shape[3]
    
    # Try typical patch sizes
    possible_patch_sizes = [14, 16]
    grid_h, grid_w = 0, 0
    patch_size = 0
    tokens = None
    
    for ps in possible_patch_sizes:
        gh = H_img // ps
        gw = W_img // ps
        num_patches = gh * gw
        if N >= num_patches:
            extra = N - num_patches
            # Assume strict match or small extra
            if extra <= N * 0.1: # Heuristic: extra tokens shouldn't be majority
                print(f"Inferred patch size {ps}: grid {gh}x{gw} = {num_patches} patches. Extra tokens: {extra}")
                grid_h, grid_w = gh, gw
                patch_size = ps
                tokens = features[:, extra:, :]
                break
    
    if tokens is None:
        raise ValueError(f"Could not infer patch size for feature count {N} and image size {H_img}x{W_img}")

    # Reshape for AnyUp: (B, C, H, W)
    # Verify sequence length matches grid
    if tokens.shape[1] != grid_h * grid_w:
         raise ValueError(f"Token count mismatch after trimming: {tokens.shape[1]} vs grid {grid_h}*{grid_w}")
         
    lr_features = tokens.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2).contiguous()
    print(f"LR features shape: (B, C, h, w) = {lr_features.shape}")

    # 5. AnyUp
    print("Loading AnyUp...")
    # AnyUp expects 'hr_image' of shape (B, 3, H, W).
    # Based on the notebook, it seems to work fine with standard precision or when inputs match.
    # The error suggests AnyUp weights are FloatTensor (float32) but we passed HalfTensor (float16).
    # So we should cast our inputs to float32 for AnyUp.
    
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    
    # Ensure inputs are float32 for AnyUp
    # We move them to CPU if we are OOM on GPU, or just try to free up more space.
    # Let's try to keep them on GPU if possible, but if OOM, we might need to do AnyUp on CPU.
    
    # Strategy: try GPU with small chunk. If fail, CPU.
    try:
        lr_features_f32 = lr_features.float()
        pixel_values_f32 = pixel_values.float()
        
        print("Running AnyUp on GPU...")
        with torch.no_grad():
            hr_features = upsampler(pixel_values_f32, lr_features_f32, q_chunk_size=128) # Very small chunk size
        print("AnyUp GPU finished.")
            
    except torch.cuda.OutOfMemoryError:
        print("OOM on GPU for AnyUp. Switching entire AnyUp process to CPU...")
        torch.cuda.empty_cache()
        upsampler.cpu()
        pixel_values_f32 = pixel_values_f32.cpu()
        lr_features_f32 = lr_features_f32.cpu()
        
        with torch.no_grad():
             print("Calling AnyUp (CPU fallback) with chunks...")
             hr_features = upsampler(pixel_values_f32, lr_features_f32, q_chunk_size=128)
             print("AnyUp CPU fallback finished.")
             
    print(f"HR features shape: {hr_features.shape}")
    
    # Move HR features to CPU immediately to free device memory (if on GPU) for PCA
    hr_features = hr_features.cpu()
    if 'lr_features' in locals():
        lr_features = lr_features.cpu()
    torch.cuda.empty_cache()

    # 6. Joint PCA Visualization
    print("Performing Joint PCA...")
    with torch.no_grad():
        # Flatten features
        # lr_features: (B, C, h, w) -> (h*w, C)
        # hr_features: (B, C, H, W) -> (H*W, C)
        
        lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C).cpu()
        hr_flat = hr_features[0].permute(1, 2, 0).reshape(-1, C).cpu()
        
        # Subsample HR features for PCA calculation to save memory/time if needed
        # But let's try full first or strict subset
        
        # Concatenate for joint PCA
        # Note: hr_flat can be huge. Let's downsample for PCA fit if it's too large.
        if hr_flat.shape[0] > 100000:
             idx = torch.randperm(hr_flat.shape[0])[:100000]
             pca_train_data = torch.cat([lr_flat, hr_flat[idx]], dim=0)
        else:
             pca_train_data = torch.cat([lr_flat, hr_flat], dim=0)
             
        # Center data
        mean = pca_train_data.mean(dim=0, keepdim=True)
        # Verify shape comp
        X_train = pca_train_data - mean
        
        # PCA via SVD
        print("Computing SVD...")
        try:
             # Use float32 for SVD
            U, S, Vh = torch.linalg.svd(X_train.float(), full_matrices=False)
            pcs = Vh[:3].T # (C, 3)
        except Exception as e:
            print(f"SVD failed checking error: {e}")
            # Fallback to sklearn if torch svd fails or OOM
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(X_train.numpy())
            pcs = torch.from_numpy(pca.components_.T).float()
            
        print("Projecting features...")
        # Project LR
        lr_centered = lr_flat - mean
        proj_lr = lr_centered @ pcs # (N_lr, 3)
        
        # Project HR
        hr_centered = hr_flat - mean
        proj_hr = hr_centered @ pcs # (N_hr, 3)
        
        # Min-Max normalize jointly to [0, 1]
        proj_all = torch.cat([proj_lr, proj_hr], dim=0)
        p_min = proj_all.min(dim=0, keepdim=True)[0]
        p_max = proj_all.max(dim=0, keepdim=True)[0]
        
        proj_lr = (proj_lr - p_min) / (p_max - p_min)
        proj_hr = (proj_hr - p_min) / (p_max - p_min)
        
        
        
        # Reshape to images
        vis_lr_radio = proj_lr.reshape(grid_h, grid_w, 3).numpy()
        vis_hr_radio = proj_hr.reshape(H_img, W_img, 3).numpy()

    # --- DINOv3 Section ---
    print("\n--- Processing DINOv3 ---")
    
    # Aggressive cleanup before DINOv3
    print("Cleaning up memory before DINOv3...")
    if 'model' in locals(): del model
    if 'image_processor' in locals(): del image_processor
    if 'lr_features' in locals(): del lr_features
    if 'hr_features' in locals(): del hr_features
    if 'upsampler' in locals(): del upsampler
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload original image to ensure clean state
    image = Image.open(urlopen(image_url)).convert('RGB')
    
    # Create DINOv3 model
    # Initialize variables to avoid UnboundLocalError
    h_dino = 0
    w_dino = 0
    vis_lr_dino = None
    vis_hr_dino = None
    
    # Try the exact name from the notebook first
    dinov3_model_name = 'vit_base_patch16_dinov3.lvd1689m'
    print(f"Loading DINOv3 model: {dinov3_model_name}...")
    
    try:
        try:
            model_dino = timm.create_model(
                dinov3_model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True
            )
        except Exception as e:
            print(f"Failed to load {dinov3_model_name}: {e}. Trying generic DINOv2 fallback...")
            # Fallback to a widely available DINOv2 model if DINOv3 fails
            dinov3_model_name = 'vit_large_patch14_dinov2.lvd142m'
            model_dino = timm.create_model(
                dinov3_model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True
            )
        except Exception as e:
            print(f"Failed to load {dinov3_model_name}: {e}. Trying vit_base...")
            dinov3_model_name = 'vit_base_patch14_dinov3.lvd142m'
            model_dino = timm.create_model(
                dinov3_model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True
            )
            
        model_dino.eval().to(device)
        if device == "cuda":
            model_dino.half()
            
        # Get data config for transform
        data_config = timm.data.resolve_model_data_config(model_dino)
        # Create transforms: resize to target_size, normalize
        transform_dino = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=data_config['mean'], std=data_config['std'])
        ])
        
        input_dino = transform_dino(image).unsqueeze(0).to(device)
        if device == "cuda":
            input_dino = input_dino.half()
            
        print("Running DINOv3 inference...")
        with torch.no_grad():
            # forward_features returns (B, N, C)
            features_dino = model_dino.forward_features(input_dino)
            
        print(f"DINOv3 features shape: {features_dino.shape}")
        
        # Parse tokens
        # DINOv3 (ViT) usually has CLS token and maybe registers.
        # Patch size 14
        # Patch size - try to get from model or infer
        try:
             patch_size_dino = model_dino.patch_embed.patch_size[0]
        except:
             # Fallback logic based on name
             if 'patch16' in dinov3_model_name:
                 patch_size_dino = 16
             elif 'patch14' in dinov3_model_name:
                 patch_size_dino = 14
             else:
                 patch_size_dino = 14 # Default assumption
                 
        print(f"Using DINOv3 patch size: {patch_size_dino}")
        
        h_dino = target_size[0] // patch_size_dino
        w_dino = target_size[1] // patch_size_dino
        num_patches_dino = h_dino * w_dino
        
        N_dino = features_dino.shape[1]
        extra_dino = N_dino - num_patches_dino
        
        if extra_dino > 0:
            print(f"Removing {extra_dino} extra tokens from DINOv3 features.")
            # CLS and registers are typically at the beginning or end. 
            # In timm ViT, CLS is at index 0. Registers might follow.
            # DINOv3: often CLS (1) + Registers (4).
            tokens_dino = features_dino[:, extra_dino:, :]
        else:
            tokens_dino = features_dino
            
        # Reshape for AnyUp: (B, C, h, w)
        C_dino = tokens_dino.shape[2]
        lr_features_dino = tokens_dino.reshape(1, h_dino, w_dino, C_dino).permute(0, 3, 1, 2).contiguous()
        
        # Run AnyUp for DINOv3
        print("Running AnyUp for DINOv3...")
        # Prepare input for AnyUp (needs to be float32 probably for safety, or half if on GPU)
        # We need 'pixel_values' equivalent for DINOv3. 
        # AnyUp expects the HR image tensor. We can reuse 'input_dino', but AnyUp expects 
        # specific normalization (ImageNet). DINOv3 uses ImageNet norm too.
        
        # Re-load AnyUp if it was cleared
        upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
        # Ensure inputs are float32 for AnyUp, matching RADIO strategy
        
        try:
            with torch.no_grad():
                # Cast inputs to float32
                input_dino_f32 = input_dino.float()
                lr_features_dino_f32 = lr_features_dino.float()
                hr_features_dino = upsampler(input_dino_f32, lr_features_dino_f32, q_chunk_size=512)
        except torch.cuda.OutOfMemoryError:
             print("OOM on GPU for DINOv3 AnyUp. Using CPU fallback.")
             torch.cuda.empty_cache()
             upsampler.cpu()
             input_dino_cpu = input_dino.float().cpu()
             lr_features_dino_cpu = lr_features_dino.float().cpu()
             with torch.no_grad():
                 hr_features_dino = upsampler(input_dino_cpu, lr_features_dino_cpu, q_chunk_size=256)
             hr_features_dino = hr_features_dino.to(device) # Move back if possible? No, keep CPU for PCA.
             if device == "cuda":
                 hr_features_dino = hr_features_dino.cpu()

        # PCA for DINOv3
        print("Computing PCA for DINOv3...")
        lr_flat_dino = lr_features_dino[0].permute(1, 2, 0).reshape(-1, C_dino).float().cpu()
        hr_flat_dino = hr_features_dino[0].permute(1, 2, 0).reshape(-1, C_dino).float().cpu()
        
        # Subsample for PCA
        if hr_flat_dino.shape[0] > 100000:
             idx = torch.randperm(hr_flat_dino.shape[0])[:100000]
             pca_train_dino = torch.cat([lr_flat_dino, hr_flat_dino[idx]], dim=0)
        else:
             pca_train_dino = torch.cat([lr_flat_dino, hr_flat_dino], dim=0)
             
        mean_dino = pca_train_dino.mean(dim=0, keepdim=True)
        X_dino = pca_train_dino - mean_dino
        
        U, S, Vh = torch.linalg.svd(X_dino, full_matrices=False)
        pcs_dino = Vh[:3].T
        
        proj_lr_dino = (lr_flat_dino - mean_dino) @ pcs_dino
        proj_hr_dino = (hr_flat_dino - mean_dino) @ pcs_dino
        
        proj_all_dino = torch.cat([proj_lr_dino, proj_hr_dino], dim=0)
        p_min_d = proj_all_dino.min(dim=0, keepdim=True)[0]
        p_max_d = proj_all_dino.max(dim=0, keepdim=True)[0]
        
        proj_lr_dino = (proj_lr_dino - p_min_d) / (p_max_d - p_min_d)
        proj_hr_dino = (proj_hr_dino - p_min_d) / (p_max_d - p_min_d)
        
        vis_lr_dino = proj_lr_dino.reshape(h_dino, w_dino, 3).numpy()
        # hr_features_dino shape might be different? It matches input_dino spatial dim (target_size)
        vis_hr_dino = proj_hr_dino.reshape(target_size[0], target_size[1], 3).numpy()
        
    except Exception as e:
        print(f"DINOv3 processing failed: {e}")
        import traceback
        traceback.print_exc()
        vis_lr_dino = np.zeros_like(vis_lr_radio)
        vis_hr_dino = np.zeros_like(vis_hr_radio)

    # 7. Save Plots
    print("Saving plots...")
    # Layout: 1 row, 5 cols: Original, RADIO LR, RADIO HR, DINO LR, DINO HR
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Original Image
    axes[0].imshow(image.resize(target_size)) # Show resized version used for inf
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # RADIO
    axes[1].imshow(vis_lr_radio)
    axes[1].set_title(f"RADIO v4 LR\n{grid_h}x{grid_w}")
    axes[1].axis('off')
    
    axes[2].imshow(vis_hr_radio)
    axes[2].set_title(f"RADIO v4 HR (AnyUp)\n{H_img}x{W_img}")
    axes[2].axis('off')
    
    # DINOv3
    if vis_lr_dino is not None:
        axes[3].imshow(vis_lr_dino)
        axes[3].set_title(f"DINO LR\n{h_dino}x{w_dino}")
    else:
        axes[3].text(0.5, 0.5, "DINOv3 Failed", ha='center', va='center')
        axes[3].set_title("DINO LR (Failed)")
    axes[3].axis('off')

    if vis_hr_dino is not None:
        axes[4].imshow(vis_hr_dino)
        axes[4].set_title(f"DINO HR (AnyUp)\n{target_size[0]}x{target_size[1]}")
    else:
        axes[4].text(0.5, 0.5, "DINOv3 Failed", ha='center', va='center')
        axes[4].set_title("DINO HR (Failed)")
    axes[4].axis('off')
    
    plt.tight_layout()
    out_path = r"c:\Users\coach\myfiles\postdoc2\code\plots\radio_dinov3_anyup_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    import sys
    try:
        visualize_radio()
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
