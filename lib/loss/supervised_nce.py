import torch


def supervised_infonce_loss(F, L, temperature=0.1, EPS=1e-5, reduction="mean"):
    """
    Compute the InfoNCE loss for binary classification in a supervised manner.

    Args:
    - F: A tensor of shape [B, N, C] representing the batch of normalized feature vectors for N samples.
    - L: A tensor of shape [B, N] representing the binary labels (0 or 1) for the samples.
    - temperature: A scalar controlling the softness of the similarity function.

    Returns:
    - loss: The InfoNCE loss for the batch.
    """
    # Get batch size B and number of samples N
    B, N, C = F.shape

    # Normalize the features (assuming they are already normalized, but to be sure)
    # F = F / F.norm(p=2, dim=-1, keepdim=True)

    # Compute pairwise cosine similarities: [B, N, N]
    similarity_matrix = torch.matmul(F, F.transpose(1, 2))  # Shape: [B, N, N]

    # Mask to avoid using self-similarity (i.e., not comparing each sample with itself)
    mask = torch.eye(N, device=F.device).bool().unsqueeze(0)  # Shape: [1, N, N]
    similarity_matrix.masked_fill_(
        mask, -float("inf")
    )  # Mask out self-similarities (set them to -inf)

    # Construct positive/negative pair mask based on labels: [B, N, N]
    labels = L.unsqueeze(1) == L.unsqueeze(
        2
    )  # Shape: [B, N, N] (True for positive pairs, False for negative pairs)
    # print(L.shape,labels.shape,labels.float().sum(),labels.float().mean())

    # Compute the denominator of the InfoNCE loss
    exp_sim = torch.exp(similarity_matrix / temperature)  # Shape: [B, N, N]
    exp_sim_sum = exp_sim.sum(dim=-1, keepdim=True)  # Shape: [B, N, 1]

    # Compute the numerator: similarity of positive pairs
    positive_exp_sim = (
        exp_sim * labels.float()
    )  # Shape: [B, N, N] (Only keep positive pair similarities)
    positive_sim = positive_exp_sim.sum(
        dim=-1, keepdim=True
    )  # Sum across each row, i.e., for each sample: [B, N, 1]

    # print(exp_sim_sum, positive_sim)

    # Compute the InfoNCE loss: -log(positive_sim / exp_sim_sum)
    loss = torch.log(positive_sim / (exp_sim_sum + EPS) + EPS)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass

    return loss


import torch.nn.functional as F

if __name__ == "__main__":
    # Example batch of normalized features and labels
    B, N, C = 2, 5, 128  # Batch size, number of samples, feature dimension
    f = F.normalize(torch.randn(B, N, C), dim=-1)  # Generate normalized feature tensor
    L = torch.randint(0, 2, (B, N))  # Binary labels for each sample
    print(f, L)

    # Compute InfoNCE loss
    loss = supervised_infonce_loss(f, L)
    print("InfoNCE Loss:", loss.item())
