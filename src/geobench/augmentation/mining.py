"""FP32 reference embeddings, centroids and exact-tie confuser selection."""
import torch


def mine_confusers(features, labels):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    if (features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels)
            or len(features) == 0 or labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool):
        raise ValueError('invalid reference features or labels')
    if not torch.isfinite(features).all() or (labels < 0).any():
        raise ValueError('invalid reference values')
    labels = labels.to(features.device)
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('zero or invalid embedding norm')
    embeddings = features / norms
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError('at least two countries are required')
    centroids = torch.stack([embeddings[labels == class_id].mean(dim=0) for class_id in classes])
    centroid_norms = torch.linalg.vector_norm(centroids, dim=1, keepdim=True)
    if not torch.isfinite(centroid_norms).all() or (centroid_norms == 0).any():
        raise ValueError('zero or invalid centroid norm')
    normalized = centroids / centroid_norms
    similarities = normalized @ normalized.T
    if not torch.isfinite(similarities).all():
        raise ValueError('invalid cosine similarity')
    confusers = {}
    for i, class_id in enumerate(classes):
        candidates = [(float(similarities[i,j]), other) for j,other in enumerate(classes) if j != i]
        confusers[class_id] = min(candidates, key=lambda candidate: (-candidate[0],candidate[1]))[1]
    return {'embeddings': embeddings, 'centroids': centroids, 'similarities': similarities,
            'class_ids': classes, 'confusers': confusers}
