import hashlib

from ...assets import load_archive


def make_da_hgrid(bundle, nu, latent_structure, nearest_horizontal, nearest_vertical, batch, **_):
    """Load a precomputed data-assimilation attention grid."""
    identity = (nu, latent_structure, nearest_horizontal, nearest_vertical, batch)
    checksum = hashlib.md5(str(identity).encode(), usedforsecurity=False).hexdigest()[:16]
    path = bundle.global_path(f"meshes/hgrid_da_{checksum}.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Required data-assimilation grid is missing: {path}")
    return load_archive(path)
