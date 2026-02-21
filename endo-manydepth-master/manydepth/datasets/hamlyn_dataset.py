from __future__ import absolute_import, division, print_function

"""
Dataset loader for the Hamlyn endoscopic video dataset.

This implementation extends the generic :class:`MonoDataset` to load
monocular sequences from the Hamlyn dataset.  Each sample in the Hamlyn
split file is expected to be a path relative to the root of the dataset
(`self.data_path`) pointing to a single frame without file extension,
followed by an integer index and a side indicator (``l`` or ``r``).  The
frame index is preserved for compatibility with the common dataset API,
but the file name itself is used to load the image.  The dataset also
provides an intrinsic matrix scaled to the requested resolution.

Depth maps are optional.  When ``self.load_depth`` is enabled, the
``get_depth`` method attempts to locate a corresponding depth file in
``depth01`` for the left camera or ``depth02`` for the right camera.
Supported depth file extensions include ``.png``, ``.jpg``, ``.jpeg`` and
``.tiff``.  Depth values are returned as floats in metres (assuming the
dataset stores depth in millimetres).

The intrinsic matrix here is adapted from values reported in the
literature for Hamlyn endoscopic sequences.  It is normalised for
images resized to 320×256; the loader will rescale the intrinsics
automatically in :meth:`load_intrinsics` of the base class.
"""

import os
import numpy as np
import cv2
from PIL import Image as pil

# Import MonoDataset from the top-level module
from datasets.mono_dataset import MonoDataset


class HamlynDataset(MonoDataset):
    """Dataset loader for the Hamlyn endoscopic sequences with dynamic intrinsics and
    neighbour snapping.

    Each sequence has its own ``intrinsics.txt`` file located in the second-level
    directory (e.g. ``rectified01/rectified01/intrinsics.txt``).  At runtime
    the loader reads this file and normalises the intrinsics to the current
    image dimensions.  If a requested neighbouring frame is missing, the loader
    searches for the closest available frame index within a small window and
    returns that image instead (neighbour snapping).
    """

    def __init__(self, *args, **kwargs):
        super(HamlynDataset, self).__init__(*args, **kwargs)
        # Default normalised intrinsic matrix for Hamlyn sequences.  This will
        # be used as a fallback when per-sequence intrinsics are unavailable.
        self.K = np.array([
            [1.2270, 0.0,    0.5296, 0.0],
            [0.0,    1.2067, 0.4012, 0.0],
            [0.0,    0.0,    1.0,    0.0],
            [0.0,    0.0,    0.0,    1.0]
        ], dtype=np.float32)
        # Cache for per-sequence intrinsics to avoid re-reading the file on every call
        self._intrinsic_cache = {}
        # Map side indicators to indices (unused here but provided for completeness)
        self.side_map = {"l": 2, "r": 3}

    def check_depth(self):
        # Depth supervision is disabled by default for Hamlyn sequences.  Override
        # this method if you wish to enable loading ground truth depth maps.
        return False

    def index_to_folder_and_frame_idx(self, index):
        """
        Parse a line from ``self.filenames``.  Each line should have the form

            ``<relative/path/to/frame> <frame_index> <side>``

        Only ``folder`` (relative path) is needed to load the image.  ``frame_index``
        and ``side`` are preserved for API compatibility.
        """
        parts = self.filenames[index].split()
        folder = parts[0]
        frame_index = int(parts[1]) if len(parts) >= 2 else 0
        side = parts[2] if len(parts) >= 3 else None
        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        """
        Compose the absolute path to an image, including the frame index.  The
        ``folder`` argument comes directly from the splits file and may be
        provided in a number of formats, including:

            ``rectified08/rectified08/image01`` (full path)
            ``rectified08/rectified08`` (sequence without specifying camera)
            ``rectified08`` (only sequence name)

        This method resolves the appropriate camera subdirectory based on the
        provided ``side`` (left/right) and returns the fully qualified path
        within ``self.data_path``.  Neighbour snapping in :meth:`get_color`
        will handle missing files.
        """
        # Determine the appropriate camera subdirectory
        def _resolve_image_folder(base: str, side_flag: str) -> str:
            parts = base.split('/')
            # If the last part already specifies an image subfolder (image01/image02), return as is
            if parts[-1].lower().startswith('image'):
                return base
            # If two parts are provided (e.g. rectified08/rectified08), append the camera
            if len(parts) >= 2 and parts[1].lower().startswith('rectified'):
                cam = 'image01' if (side_flag in ['l', 'L']) else 'image02'
                return os.path.join(base, cam)
            # If only the sequence name is provided, duplicate it and append the camera
            if len(parts) == 1:
                seq = parts[0]
                cam = 'image01' if (side_flag in ['l', 'L']) else 'image02'
                return os.path.join(seq, seq, cam)
            # Fallback: return the original folder
            return base

        frame_str = f"{frame_index:010d}"
        image_folder = _resolve_image_folder(folder, side or '')
        return os.path.join(self.data_path, image_folder, frame_str + self.img_ext)

    def _find_neighbour_index(self, folder: str, frame_index: int, side: str, max_offset: int = 5) -> int:
        """
        Attempt to locate the closest existing frame index if the requested
        ``frame_index`` does not correspond to an existing file.  The search
        iterates over offsets ±1, ±2, … up to ``max_offset`` and returns
        the index of the first existing image.  If no neighbour is found, the
        original index is returned and the caller may raise an exception.
        """
        for offset in range(1, max_offset + 1):
            # Search previous then next
            for sign in (-1, 1):
                neighbour = frame_index + sign * offset
                if neighbour < 0:
                    continue
                candidate_path = self.get_image_path(folder, neighbour, side)
                if os.path.exists(candidate_path):
                    return neighbour
        # Fallback to original index (will likely raise later if file does not exist)
        return frame_index

    def get_color(self, folder, frame_index, side, do_flip):
        """
        Load an RGB image for the given ``folder`` and ``frame_index``.  If the
        requested frame does not exist on disk, neighbour snapping is performed
        to locate the closest available frame index within a small window.  The
        loaded image is optionally flipped horizontally if ``do_flip`` is True.
        """
        # Compute the nominal path
        image_path = self.get_image_path(folder, frame_index, side)
        # If the file is missing, find the closest neighbour
        if not os.path.exists(image_path):
            snapped_index = self._find_neighbour_index(folder, frame_index, side)
            image_path = self.get_image_path(folder, snapped_index, side)
        # Load the image
        color = self.loader(image_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def load_intrinsics(self, folder, frame_index):
        """
        Load and normalise the camera intrinsic matrix for the given sequence.

        The Hamlyn dataset provides a per-sequence ``intrinsics.txt`` file
        located at ``<sequence>/<sequence>/intrinsics.txt``.  Each file stores
        a 3×4 matrix in pixel units.  This method reads the matrix, extracts
        the focal lengths and principal point, and converts them into a normalised
        4×4 intrinsic matrix compatible with the scaling performed in
        :class:`MonoDataset`.  If the file cannot be read, the default
        normalised matrix from ``self.K`` is returned.
        """
        # Derive the path to the intrinsics file by taking the first two path
        # components of ``folder`` (e.g. rectified01/rectified01/image01 -> rectified01/rectified01)
        parts = folder.split('/')
        # Derive the base directory that contains the intrinsics file.  If
        # fewer than two parts are provided (e.g. ``rectified08``), duplicate
        # the sequence name to form ``rectified08/rectified08``.  Otherwise,
        # take the first two path segments.
        if len(parts) >= 2:
            base_dir = os.path.join(self.data_path, parts[0], parts[1])
        elif len(parts) == 1:
            base_dir = os.path.join(self.data_path, parts[0], parts[0])
        else:
            # Should not happen, but return default if it does
            return self.K.copy()
        intr_path = os.path.join(base_dir, "intrinsics.txt")
        # Return cached intrinsics if available
        if intr_path in self._intrinsic_cache:
            return self._intrinsic_cache[intr_path].copy()
        if os.path.exists(intr_path):
            try:
                intr = np.loadtxt(intr_path)
                # intr may be 3x4; extract fx, fy, cx, cy from the first two rows
                if intr.ndim == 2 and intr.shape[0] >= 2:
                    fx = float(intr[0, 0])
                    cx = float(intr[0, 2])
                    fy = float(intr[1, 1])
                    cy = float(intr[1, 2])
                    # Determine the original image size by reading the current frame
                    img_w, img_h = None, None
                    try:
                        # Build path to the current image.  Pass an empty side flag
                        sample_path = self.get_image_path(folder, frame_index, '')
                        with open(sample_path, 'rb') as f:
                            im = pil.open(f)
                            img_w, img_h = im.size
                    except Exception:
                        # Fall back to dataset target size if image cannot be read
                        img_w, img_h = self.width, self.height
                    # Normalise intrinsics
                    K_norm = np.array([
                        [fx / img_w, 0.0, cx / img_w, 0.0],
                        [0.0, fy / img_h, cy / img_h, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    self._intrinsic_cache[intr_path] = K_norm
                    return K_norm.copy()
            except Exception:
                # Fall back to default if reading fails
                pass
        # Default behaviour: return the original normalised K
        return self.K.copy()

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        Load the ground truth depth map (if available) for the given frame.  Depth
        values are assumed to be stored in millimetres and are converted to
        metres.  If ``do_flip`` is True, the depth map is flipped horizontally.
        """
        parts = folder.split('/')
        if len(parts) < 3:
            raise ValueError(f"Unexpected folder format: {folder}")
        base_dir = os.path.join(self.data_path, parts[0], parts[1])
        file_name = parts[-1]
        depth_dir = "depth01" if (side in ['l', 'L']) else "depth02"
        depth_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            candidate = os.path.join(base_dir, depth_dir, file_name + ext)
            if os.path.exists(candidate):
                depth_path = candidate
                break
        if depth_path is None:
            raise FileNotFoundError(f"Could not find depth map for {folder} with side {side}")
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # Convert mm to m if values are large
        if depth_gt.max() > 0:
            depth_gt = depth_gt / 1000.0
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt