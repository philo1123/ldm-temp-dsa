import torchvision.transforms as transforms

def create_transform_pipeline(size, augmentation_config=None):
    """
    Create a transformation pipeline for image preprocessing.
    
    Args:
        size (int): Target size for resizing the image (width = height = size).
        augmentation_config (dict, optional): A dictionary specifying which augmentation transforms to enable.
            Example:
            {
              "rotation": True,
              "rotation_degrees": 10,
              "color_jitter": True,
              "brightness": 0.2,
              "contrast": 0.2,
              "horizontal_flip": False
            }
    
    Returns:
        transforms.Compose: A composed transform pipeline.
    """
    transform_list = []
    # Convert to grayscale always
    transform_list.append(transforms.Lambda(lambda img: img.convert("L")))
    # Center crop to the shortest side
    transform_list.append(transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))))
    
    # Apply augmentations if a config is provided
    if augmentation_config is not None:
        if augmentation_config.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if augmentation_config.get("rotation", False):
            degrees = augmentation_config.get("rotation_degrees", 10)
            transform_list.append(transforms.RandomRotation(degrees=degrees))
        if augmentation_config.get("color_jitter", False):
            brightness = augmentation_config.get("brightness", 0.2)
            contrast = augmentation_config.get("contrast", 0.2)
            transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
        
    # Resize to target size
    if size is not None:
        transform_list.append(transforms.Resize((size, size)))
    # Convert to tensor and normalize to [-1, 1]
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    return transforms.Compose(transform_list)