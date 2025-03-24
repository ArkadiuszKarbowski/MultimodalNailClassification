import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(dataset_stats, modality, training=False):
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

    resize = A.Compose(
        [A.Resize(height=target_height, width=target_width, p=1.0)],
        additional_targets={"uv": "image"},
    )

    normalize_normal = A.Compose(
        [
            A.Normalize(
                mean=dataset_stats["mean_normal"],
                std=dataset_stats["std_normal"],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    normalize_uv = A.Compose(
        [
            A.Normalize(
                mean=dataset_stats["mean_uv"],
                std=dataset_stats["std_uv"],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    if not training:
        return {
            "shared_transform": resize,
            "normal_transform": normalize_normal,
            "uv_transform": normalize_uv,
        }

    geometric = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.85, 1.15),
                rotate=(-25, 25),
                shear=(-5, 5),
                p=0.7,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.RandomResizedCrop(
                size=(target_height, target_width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        ],
        additional_targets={"uv": "image"},
    )

    color_normal = A.Compose(
        [
            A.OneOf(
                [
                    A.CLAHE(clip_limit=3, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0
                    ),
                ],
                p=0.7,
            ),
            A.RandomGamma(gamma_limit=(90, 110)),
        ]
    )

    color_uv = A.Compose(
        [
            A.OneOf(
                [
                    A.CLAHE(clip_limit=3, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=0,
                    ),
                ],
                p=0.7,
            ),
            A.RandomGamma(gamma_limit=(90, 110)),
        ]
    )

    transformation_shared = A.Compose(
        [
            resize,
            geometric,
        ]
    )

    transformation_normal = A.Compose([color_normal, normalize_normal])

    transformation_uv = A.Compose([color_uv, normalize_uv])
    return {
        "shared_transform": transformation_shared,
        "normal_transform": transformation_normal,
        "uv_transform": transformation_uv,
    }
