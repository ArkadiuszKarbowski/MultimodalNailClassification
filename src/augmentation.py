import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(dataset_stats):
    transform_normal = A.Compose(
        [
            A.Resize(
                dataset_stats["resize_shape"][0], dataset_stats["resize_shape"][1]
            ),
            A.Normalize(
                mean=dataset_stats["mean_normal"],
                std=dataset_stats["std_normal"],
            ),
            ToTensorV2(),
        ]
    )

    transform_uv = A.Compose(
        [
            A.Resize(
                dataset_stats["resize_shape"][0], dataset_stats["resize_shape"][1]
            ),
            A.Normalize(
                mean=dataset_stats["mean_uv"],
                std=dataset_stats["std_uv"],
            ),
            ToTensorV2(),
        ]
    )

    return {"normal": transform_normal, "uv": transform_uv}
