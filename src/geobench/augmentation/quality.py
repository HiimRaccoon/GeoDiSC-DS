"""Exact-duplicate checks common to generated training images."""


class HoldoutCollision(ValueError):
    """A candidate matched the real validation or test set; freeze is blocked."""


class TechnicalRejection(ValueError):
    """A candidate has a numerical or technical failure eligible for retry."""


def validate_candidate_tensor(candidate, expected_size):
    import torch
    if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (3, expected_size[1], expected_size[0]):
        return 'geometry_mismatch'
    if not torch.isfinite(candidate).all():
        return 'nonfinite_pixels'
    if (candidate < 0).any() or (candidate > 1).any():
        return 'pixel_range'
    return None


def audit_candidate(file_hash, pixel_hash, *, train_files, train_pixels, holdout_files, holdout_pixels, accepted_files, accepted_pixels):
    if not isinstance(file_hash, str) or not file_hash or not isinstance(pixel_hash, str) or not pixel_hash:
        raise ValueError('candidate requires file and decoded pixel hashes')
    if file_hash in holdout_files or pixel_hash in holdout_pixels:
        raise HoldoutCollision('candidate matches a holdout image')
    if file_hash in train_files or pixel_hash in train_pixels:
        return 'training_exact_duplicate'
    if file_hash in accepted_files or pixel_hash in accepted_pixels:
        return 'pool_exact_duplicate'
    return None
