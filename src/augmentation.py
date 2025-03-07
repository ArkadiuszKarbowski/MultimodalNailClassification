import albumentations as A
from albumentations.pytorch import ToTensorV2


class NailAugmentation:
    def __init__(self, mean, std, p=0.5):
        self.transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1),
                        A.ColorJitter(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=1),
                        A.MotionBlur(p=1),
                        A.MedianBlur(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(p=1),
                        A.GridDistortion(p=1),
                        A.OpticalDistortion(p=1),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=1),
                        A.ShiftScaleRotate(p=1),
                    ],
                    p=0.5,
                ),
                A.Resize(224, 224),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            p=p,
        )
