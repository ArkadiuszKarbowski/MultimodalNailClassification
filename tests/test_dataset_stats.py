import pytest
import numpy as np
import torch
from torchvision.io import write_png

from src.utils.prepare_dataset_utils.dataset_stats import (
    ImageDataset,
    calculate_dataset_statistics,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_images(temp_dir):
    # Create sample UV and normal images
    img_dir = temp_dir / "images"
    img_dir.mkdir()

    # Create 5 UV images (all 0.5 values)
    uv_paths = []
    for i in range(5):
        path = img_dir / f"UV_{i}.png"
        img = torch.full((3, 256, 256), 127, dtype=torch.uint8)  # 127/255 \u2248 0.498
        write_png(img, str(path))
        uv_paths.append(str(path))

    # Create 5 normal images (all 0.8 values)
    normal_paths = []
    for i in range(5):
        path = img_dir / f"normal_{i}.png"
        img = torch.full((3, 300, 400), 204, dtype=torch.uint8)  # 204/255 \u2248 0.8
        write_png(img, str(path))
        normal_paths.append(str(path))

    return uv_paths + normal_paths


def test_image_dataset(sample_images):
    dataset = ImageDataset(sample_images)
    assert len(dataset) == len(sample_images)

    for i in range(len(sample_images)):
        img, valid = dataset[i]
        assert valid is True
        assert img.shape == (3, 512, 512)


def test_calculate_statistics(sample_images, temp_dir):
    stats = calculate_dataset_statistics(
        sample_images,
        resize_shape=(512, 512),
        batch_size=4,
        use_gpu=False,
        output_json=str(temp_dir / "stats.json"),
        verbose=False,
    )

    assert "mean_uv" in stats
    assert "std_uv" in stats
    assert np.allclose(stats["mean_uv"], [0.498] * 3, atol=0.01)


def test_empty_dataset(temp_dir):
    stats = calculate_dataset_statistics(
        [],
        output_json=str(temp_dir / "empty_stats.json"),
    )
    assert stats["std_uv"] == [1.0, 1.0, 1.0]


def test_image_resizing(mocker, temp_dir):
    img_path = temp_dir / "test_img.png"
    img = torch.randint(0, 255, (3, 100, 200), dtype=torch.uint8)
    write_png(img, str(img_path))

    mocker.patch(
        "src.utils.prepare_dataset_utils.dataset_stats.read_image", return_value=img
    )

    dataset = ImageDataset([str(img_path)], resize_shape=(50, 50))
    img_tensor, valid = dataset[0]
    assert img_tensor.shape == (3, 50, 50)


def test_memory_management(sample_images):
    stats = calculate_dataset_statistics(
        sample_images,
        batch_size=2,
        use_gpu=False,
    )
    assert isinstance(stats, dict)
