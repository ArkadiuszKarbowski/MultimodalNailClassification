import pytest
import numpy as np
from concurrent.futures import Future

from src.utils.calculate_dataset_stats import (
    process_image,
    process_image_batch,
    calculate_dataset_statistics,
)


@pytest.fixture
def mock_image():
    return np.ones((100, 100, 3)) * 128


def test_process_image_valid(mocker, mock_image):
    mock_imread = mocker.patch("cv2.imread", return_value=mock_image)
    mock_cvtColor = mocker.patch("cv2.cvtColor", return_value=mock_image)

    path, mean, std = process_image("test_img.jpg")

    assert path == "test_img.jpg"
    assert np.allclose(mean, np.array([0.5, 0.5, 0.5]), rtol=1e-2)
    assert np.allclose(std, np.array([0.0, 0.0, 0.0]), atol=1e-5)


def test_process_image_invalid(mocker):
    mocker.patch("cv2.imread", return_value=None)

    path, mean, std = process_image("invalid_img.jpg")

    assert path == "invalid_img.jpg"
    assert mean is None
    assert std is None


def test_process_image_batch(mocker):
    mock_process = mocker.patch("src.utils.calculate_dataset_stats.process_image")
    mock_process.side_effect = [
        ("img1.jpg", np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])),
        ("img2.jpg", np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3])),
    ]

    results = process_image_batch(["img1.jpg", "img2.jpg"])

    assert len(results) == 2
    assert results[0][0] == "img1.jpg"
    assert np.allclose(results[0][1], np.array([0.1, 0.2, 0.3]))
    assert np.allclose(results[0][2], np.array([0.4, 0.5, 0.6]))
    assert results[1][0] == "img2.jpg"
    assert np.allclose(results[1][1], np.array([0.7, 0.8, 0.9]))
    assert np.allclose(results[1][2], np.array([0.1, 0.2, 0.3]))


def test_calculate_dataset_statistics(mocker):
    future1 = Future()
    future1.set_result(
        [
            (
                "path/to/UV_img1.jpg",
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6]),
            ),
            (
                "path/to/UV_img2.jpg",
                np.array([0.2, 0.3, 0.4]),
                np.array([0.5, 0.6, 0.7]),
            ),
        ]
    )

    future2 = Future()
    future2.set_result(
        [
            (
                "path/to/normal_img1.jpg",
                np.array([0.7, 0.8, 0.9]),
                np.array([0.2, 0.3, 0.4]),
            ),
            (
                "path/to/normal_img2.jpg",
                np.array([0.6, 0.7, 0.8]),
                np.array([0.1, 0.2, 0.3]),
            ),
        ]
    )

    mock_executor = mocker.patch(
        "src.utils.calculate_dataset_stats.ProcessPoolExecutor"
    )
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = (
        lambda func, batch: future1 if "UV" in batch[0] else future2
    )

    mocker.patch(
        "src.utils.calculate_dataset_stats.as_completed", side_effect=lambda fs: fs
    )

    stats = calculate_dataset_statistics(
        [
            "path/to/UV_img1.jpg",
            "path/to/UV_img2.jpg",
            "path/to/normal_img1.jpg",
            "path/to/normal_img2.jpg",
        ],
        batch_size=2,
    )

    # Expected values based on our mock data
    expected_mean_uv = np.array([0.15, 0.25, 0.35]).tolist()
    expected_std_uv = np.array([0.45, 0.55, 0.65]).tolist()
    expected_mean_normal = np.array([0.65, 0.75, 0.85]).tolist()
    expected_std_normal = np.array([0.15, 0.25, 0.35]).tolist()

    assert "mean_uv" in stats
    assert stats["mean_uv"] == pytest.approx(expected_mean_uv)
    assert stats["std_uv"] == pytest.approx(expected_std_uv)
    assert stats["mean_normal"] == pytest.approx(expected_mean_normal)
    assert stats["std_normal"] == pytest.approx(expected_std_normal)


def test_calculate_dataset_statistics_empty_list():
    stats = calculate_dataset_statistics([])

    assert stats["mean_uv"] == [0.0, 0.0, 0.0]
    assert stats["std_uv"] == [1.0, 1.0, 1.0]
    assert stats["mean_normal"] == [0.0, 0.0, 0.0]
    assert stats["std_normal"] == [1.0, 1.0, 1.0]
