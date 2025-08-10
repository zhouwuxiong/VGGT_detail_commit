import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
# dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#
# # Initialize the model and load the pretrained weights.
# # This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
#
# # Load and preprocess example images (replace with your own image paths)
# # image_names = ["examples/room/images/no_overlap_1.png", "examples/room/images/no_overlap_2.jpg", "examples/room/images/no_overlap_3.jpg",
# #                "examples/room/images/no_overlap_4.jpg", "examples/room/images/no_overlap_5.jpg", "examples/room/images/no_overlap_6.jpg",
# #                "examples/room/images/no_overlap_7.jpg", "examples/room/images/no_overlap_8.jpg"]
#
# image_names = ["examples/room/images/no_overlap_1.png", "examples/room/images/no_overlap_2.jpg"]
#
# images = load_and_preprocess_images(image_names).to(device)
#
# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)


def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions