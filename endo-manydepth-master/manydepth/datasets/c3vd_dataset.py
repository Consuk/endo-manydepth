from __future__ import absolute_import, division, print_function

import json
import os
import re

import numpy as np
from PIL import Image as pil

from .mono_dataset import MonoDataset


_ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_NON_COLOR_TOKENS = ("depth", "normal", "flow", "occlusion", "occ", "mask", "pose")
_COLOR_HINT_TOKENS = ("color", "rgb", "image", "img")


class C3VDDataset(MonoDataset):
    """Dataloader for C3VD / C3VD-like splits.

    Expected split format per line:
        <folder> <frame_idx> l

    The loader supports padded and unpadded frame names and prioritizes RGB
    files (e.g. ``*_color.png``) over depth/flow/occlusion assets.
    """

    def __init__(self, *args, intrinsics_file=None, **kwargs):
        super(C3VDDataset, self).__init__(*args, **kwargs)

        # Default normalized intrinsics used as a robust fallback.
        self.K = np.array([
            [0.57052094, 0.0, 0.5, 0.0],
            [0.0, 0.71185082, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.intrinsics_file = intrinsics_file
        self._intrinsics_map = self._load_intrinsics_map(intrinsics_file)
        self._frame_cache = {}

    def check_depth(self):
        return False

    def index_to_folder_and_frame_idx(self, index):
        parts = self.filenames[index].split()
        if len(parts) < 2:
            raise ValueError("Invalid C3VD split line (expected at least folder and frame): '{}'".format(self.filenames[index]))

        folder = parts[0]
        frame_index = int(parts[1])
        side = parts[2] if len(parts) > 2 else "l"
        return folder, frame_index, side

    def get_color(self, folder, frame_index, side, do_flip):
        image_path = self.get_image_path(folder, frame_index, side)
        color = self.loader(image_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        image_path = self._resolve_frame_path(folder, int(frame_index))
        if image_path is None:
            raise FileNotFoundError(
                "Could not resolve C3VD RGB frame for folder='{}' frame='{}' under '{}'".format(
                    folder, frame_index, self.data_path
                )
            )
        return image_path

    def load_intrinsics(self, folder, frame_index):
        if not self._intrinsics_map:
            return self.K.copy()

        norm_folder = folder.replace("\\", "/")
        candidates = [
            norm_folder,
            norm_folder.rstrip("/"),
            os.path.basename(norm_folder.rstrip("/")),
        ]

        for key in candidates:
            if key in self._intrinsics_map:
                return self._intrinsics_map[key].copy()

        if "default" in self._intrinsics_map:
            return self._intrinsics_map["default"].copy()

        return self.K.copy()

    def _resolve_frame_path(self, folder, frame_index):
        roots = self._candidate_roots(folder)

        # Fast path with deterministic candidate names.
        for candidate in self._enumerate_direct_candidates(roots, frame_index):
            if os.path.isfile(candidate):
                return candidate

        # Fallback path using folder-level cached discovery.
        cache_key = folder.replace("\\", "/")
        if cache_key not in self._frame_cache:
            self._frame_cache[cache_key] = self._scan_folder_for_frames(roots)

        return self._frame_cache[cache_key].get(frame_index)

    def _candidate_roots(self, folder):
        seq_root = os.path.join(self.data_path, folder)
        candidates = [
            seq_root,
            os.path.join(seq_root, "data"),
            os.path.join(seq_root, "rgb"),
            os.path.join(seq_root, "color"),
            os.path.join(seq_root, "images"),
            os.path.join(seq_root, "image"),
        ]

        roots = []
        seen = set()
        for p in candidates:
            norm = os.path.normpath(p)
            if norm not in seen:
                roots.append(norm)
                seen.add(norm)
        return roots

    def _enumerate_direct_candidates(self, roots, frame_index):
        frame_tokens = []
        raw_token = str(frame_index)
        for width in [0, 4, 5, 6]:
            token = raw_token if width == 0 else ("{:0" + str(width) + "d}").format(frame_index)
            if token not in frame_tokens:
                frame_tokens.append(token)

        suffixes = ["_color", "", "_rgb", "_image"]
        exts = [self.img_ext] + [e for e in _ALLOWED_IMAGE_EXTS if e != self.img_ext]

        for root in roots:
            for token in frame_tokens:
                for suffix in suffixes:
                    for ext in exts:
                        yield os.path.join(root, "{}{}{}".format(token, suffix, ext))

    def _scan_folder_for_frames(self, roots):
        best = {}

        for root in roots:
            if not os.path.isdir(root):
                continue

            try:
                entries = os.listdir(root)
            except OSError:
                continue

            for name in entries:
                path = os.path.join(root, name)
                if not os.path.isfile(path):
                    continue

                stem, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext not in _ALLOWED_IMAGE_EXTS:
                    continue

                stem_lower = stem.lower()
                if any(tok in stem_lower for tok in _NON_COLOR_TOKENS):
                    continue

                idx = self._extract_frame_index(stem_lower)
                if idx is None:
                    continue

                score = self._score_candidate(stem_lower, root)
                previous = best.get(idx)
                if previous is None or score > previous[0]:
                    best[idx] = (score, path)

        return {idx: pair[1] for idx, pair in best.items()}

    @staticmethod
    def _extract_frame_index(stem):
        # Typical patterns: 0005_color, 5_color, 0005, frame_0005, rgb_5
        matches = re.findall(r"(\d+)", stem)
        if not matches:
            return None

        try:
            return int(matches[-1])
        except ValueError:
            return None

    @staticmethod
    def _score_candidate(stem_lower, root):
        score = 0
        if "_color" in stem_lower:
            score += 100
        if any(tok in stem_lower for tok in _COLOR_HINT_TOKENS):
            score += 50

        root_name = os.path.basename(root).lower()
        if root_name in ("rgb", "color", "data", "images", "image"):
            score += 20

        if any(tok in stem_lower for tok in _NON_COLOR_TOKENS):
            score -= 100

        return score

    def _load_intrinsics_map(self, intrinsics_file):
        if not intrinsics_file:
            return {}

        intrinsics_file = os.path.expanduser(intrinsics_file)
        if not os.path.isfile(intrinsics_file):
            raise FileNotFoundError("C3VD intrinsics file not found: {}".format(intrinsics_file))

        _, ext = os.path.splitext(intrinsics_file)
        ext = ext.lower()

        if ext == ".json":
            with open(intrinsics_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return self._parse_intrinsics_payload(payload)

        # Fallback text format:
        #   - one line with 6 numbers: width height fx fy cx cy
        #   - optional keyed lines: <folder> width height fx fy cx cy
        mapping = {}
        with open(intrinsics_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue

                parts = re.split(r"[\s,]+", line)

                if len(parts) == 6 and self._all_numeric(parts):
                    K = self._matrix_from_wh_fx_fy_cx_cy(*map(float, parts))
                    if K is not None:
                        mapping["default"] = K
                    continue

                if len(parts) >= 7 and self._all_numeric(parts[1:7]):
                    key = parts[0].replace("\\", "/")
                    K = self._matrix_from_wh_fx_fy_cx_cy(*map(float, parts[1:7]))
                    if K is not None:
                        mapping[key] = K

        return mapping

    def _parse_intrinsics_payload(self, payload):
        mapping = {}

        if isinstance(payload, dict):
            for key, value in payload.items():
                K = self._matrix_from_entry(value)
                if K is not None:
                    mapping[key.replace("\\", "/")] = K

        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue

                folder = item.get("folder", "default")
                K = self._matrix_from_entry(item)
                if K is not None:
                    mapping[str(folder).replace("\\", "/")] = K

        return mapping

    def _matrix_from_entry(self, entry):
        if isinstance(entry, dict):
            if "K" in entry:
                K = np.array(entry["K"], dtype=np.float32)
                if K.shape == (4, 4):
                    return K
                if K.shape == (3, 3):
                    K4 = np.eye(4, dtype=np.float32)
                    K4[:3, :3] = K
                    return K4

            keys = {k.lower(): v for k, v in entry.items()}
            required = ["width", "height", "fx", "fy", "cx", "cy"]
            if all(k in keys for k in required):
                return self._matrix_from_wh_fx_fy_cx_cy(
                    float(keys["width"]),
                    float(keys["height"]),
                    float(keys["fx"]),
                    float(keys["fy"]),
                    float(keys["cx"]),
                    float(keys["cy"]),
                )

        return None

    @staticmethod
    def _matrix_from_wh_fx_fy_cx_cy(width, height, fx, fy, cx, cy):
        if width <= 0 or height <= 0:
            return None

        return np.array([
            [fx / width, 0.0, cx / width, 0.0],
            [0.0, fy / height, cy / height, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    @staticmethod
    def _all_numeric(parts):
        try:
            for p in parts:
                float(p)
            return True
        except ValueError:
            return False
