"""Tests for /volumes/{filename}/info and /volumes/{filename}/data dtype & affine contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from app import volume
from app.main import app


def _save_temp_nifti(directory: Path, name: str, arr: np.ndarray) -> Path:
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(arr, affine)
    path = directory / name
    nib.save(img, path)
    return path


class VolumeDataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        volume.load_volume_array.cache_clear()

    def test_real_ct_abdo_uses_int16_and_includes_affine(self) -> None:
        info_res = self.client.get("/volumes/CT_Abdo.nii.gz/info")
        self.assertEqual(info_res.status_code, 200)
        info = info_res.json()

        self.assertEqual(info["dtype"], "int16")
        self.assertEqual(len(info["affine"]), 4)
        self.assertEqual(len(info["affine"][0]), 4)

        nx, ny, nz = info["shape"]
        data_res = self.client.get("/volumes/CT_Abdo.nii.gz/data")
        self.assertEqual(data_res.status_code, 200)
        self.assertEqual(data_res.headers.get("x-volume-dtype"), "int16")
        self.assertEqual(len(data_res.content), nx * ny * nz * 2)

        arr = np.frombuffer(data_res.content, dtype=np.int16).reshape(
            (nx, ny, nz), order="F"
        )
        self.assertEqual(float(arr.min()), info["min_val"])
        self.assertEqual(float(arr.max()), info["max_val"])

    def test_high_range_with_decimals_uses_float32_like_ernie_t1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            expected = np.array(
                [[[0.0, 123.45], [32768.0, 65535.5]], [[10.25, 40000.75], [5.0, 1.5]]],
                dtype=np.float32,
            )
            _save_temp_nifti(tmp_dir, "ernie_sim.nii.gz", expected)

            with patch.object(volume, "DATA_DIR", tmp_dir):
                volume.load_volume_array.cache_clear()
                info_res = self.client.get("/volumes/ernie_sim.nii.gz/info")
                self.assertEqual(info_res.status_code, 200)
                info = info_res.json()
                self.assertEqual(info["dtype"], "float32")
                self.assertEqual(len(info["affine"]), 4)

                data_res = self.client.get("/volumes/ernie_sim.nii.gz/data")
                self.assertEqual(data_res.status_code, 200)
                self.assertEqual(len(data_res.content), expected.size * 4)

                recovered = np.frombuffer(data_res.content, dtype=np.float32).reshape(
                    expected.shape, order="F"
                )
                np.testing.assert_allclose(recovered, expected, rtol=1e-5)

    def test_small_range_with_decimals_uses_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            expected = np.array([[[0.0, 12.34], [5.67, 100.5]]], dtype=np.float32)
            _save_temp_nifti(tmp_dir, "decimals.nii.gz", expected)

            with patch.object(volume, "DATA_DIR", tmp_dir):
                volume.load_volume_array.cache_clear()
                info_res = self.client.get("/volumes/decimals.nii.gz/info")
                self.assertEqual(info_res.status_code, 200)
                info = info_res.json()
                self.assertEqual(info["dtype"], "float32")

    def test_whole_numbers_in_int16_range_use_int16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            expected = np.array([[[-1024, 0], [42, 3000]]], dtype=np.int16)
            _save_temp_nifti(tmp_dir, "ct_sim.nii.gz", expected)

            with patch.object(volume, "DATA_DIR", tmp_dir):
                volume.load_volume_array.cache_clear()
                info_res = self.client.get("/volumes/ct_sim.nii.gz/info")
                self.assertEqual(info_res.status_code, 200)
                info = info_res.json()
                self.assertEqual(info["dtype"], "int16")

                data_res = self.client.get("/volumes/ct_sim.nii.gz/data")
                recovered = np.frombuffer(data_res.content, dtype=np.int16).reshape(
                    expected.shape, order="F"
                )
                np.testing.assert_array_equal(recovered, expected)


if __name__ == "__main__":
    unittest.main()
