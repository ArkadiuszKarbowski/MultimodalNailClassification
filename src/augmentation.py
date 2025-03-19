import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def identity_func(image, **kwargs):
    return image

def get_transforms(dataset_stats, modality, training = False):
    if modality == 1:
        # TODO
        pass 
    elif modality == 2:
        return get_transform_mod2(dataset_stats, training)
    elif modality == 4:
        # TODO
        pass

def get_transform_mod2(dataset_stats, training):
    target_height, target_width = dataset_stats["resize_shape"]
  
    resize = A.Compose([
        A.LongestMaxSize(max_size=max(target_height, target_width)),
        A.PadIfNeeded(
            min_height=target_height,
            min_width=target_width,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.CenterCrop(height=target_height, width=target_width),])
    
    shared_transform_basic= A.Compose([
        A.Lambda(image=identity_func)
    ], additional_targets={"uv": "image"})


    transform_normal_basic = A.Compose([
        A.Normalize(
            mean=dataset_stats["mean_normal"],
            std=dataset_stats["std_normal"],
            max_pixel_value=255.0,),
        ToTensorV2()
    ])
        
    transform_uv_basic =  A.Compose([
        A.Normalize(
            mean=dataset_stats["mean_uv"],
            std=dataset_stats["std_uv"],
            max_pixel_value=255.0,),
        ToTensorV2()
    ])
    
    if not training:
        return {'resize_transform':resize, 'shared_transform':shared_transform_basic, 'normal_transform':transform_normal_basic, 'uv_transform':transform_uv_basic}
    
    geometric = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.RandomResizedCrop(
            size = (target_height, target_width),
            scale=(0.8, 1.0),
            p=0.5
        ),
    ], additional_targets={"uv": "image"})

    color_normal = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
        A.RandomGamma(p=0.5),
    ])

    color_uv = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
        A.RandomGamma(p=0.5),
    ])


    transformation_normal = A.Compose([
        color_normal,
        transform_normal_basic
    ])

    transformation_uv = A.Compose([
        color_uv,
        transform_uv_basic
    ])
    return {'resize_transform':resize, 'shared_transform':shared_transform_basic, 'normal_transform':transformation_normal
            , 'uv_transform':transformation_uv}