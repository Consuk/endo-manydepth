# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def resolve_split_dir(split, split_root=None, file_dir=None):
    """Resolve the split directory, optionally from an external split root.

    If ``split_root`` points directly to a split directory (contains
    ``*_files.txt`` files), it is returned as-is. Otherwise this function tries
    ``split_root/<split>`` and case-insensitive matches.
    """

    def _has_split_files(path):
        if not os.path.isdir(path):
            return False
        names = (
            "train_files.txt",
            "training_files.txt",
            "val_files.txt",
            "validation_files.txt",
            "test_files.txt",
        )
        return any(os.path.isfile(os.path.join(path, n)) for n in names)

    split = str(split)
    file_dir = file_dir or os.path.dirname(__file__)

    candidate_roots = []
    if split_root:
        candidate_roots.append(os.path.expanduser(split_root))
    candidate_roots.append(os.path.join(file_dir, "splits"))

    checked = []
    for root in candidate_roots:
        if not os.path.isdir(root):
            checked.append(root)
            continue

        direct_candidates = [root, os.path.join(root, split)]

        if split.lower() != split:
            direct_candidates.append(os.path.join(root, split.lower()))
        if split.upper() != split:
            direct_candidates.append(os.path.join(root, split.upper()))

        for candidate in direct_candidates:
            if _has_split_files(candidate):
                return candidate

        try:
            entries = os.listdir(root)
        except OSError:
            checked.append(root)
            continue

        for entry in entries:
            entry_path = os.path.join(root, entry)
            if entry.lower() == split.lower() and _has_split_files(entry_path):
                return entry_path

        checked.append(root)

    raise FileNotFoundError(
        "Could not resolve split '{}' from roots: {}".format(split, checked)
    )
