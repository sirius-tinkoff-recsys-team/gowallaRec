import torch
from torch.nn.utils.rnn import pad_sequence


def pad_3d_sequence(
    tokens,
    max_sent_length: int = None,
    max_sents: int = None,
    pad_in_the_end: bool = True
) -> torch.Tensor:
    """
    Perform padding for 3D Tensor
    (Batch Size x Number of sentences x Number of words in sentences).

    Parameters
    ----------
    tokens : `List[List[int]]`, required
        Nested lists of token indexes with variable
        number of sentences and number of words in sentence.
    max_sent_length : `int`, optional (default = `None`)
        Max number of words in sentence.
        If `None` number of words in sentence
        would be determined from passed `tokens`
        (equals max number of words in sentence per batch).
    max_sents : `int`, optional (default = `None`)
        Max number of sentences in one document.
        If `None` max number of sentences
        would be determined from passed `tokens`
        (equals max number of sentences per batch).
    pad_in_the_end: `bool`, optional (default = `True`)
        Whether to pad in the end of sequence or beginning.

    Returns
    -------
    `torch.Tensor`
        Padded 3D torch.Tensor.

    Examples:
    ---------
        pad_3d_sequence(
            [[[1, 2, 3], [4, 5]], [[3, 4], [7, 8, 9, 6], [1, 2, 3]]]
        )
        tensor([[[1., 2., 3., 0.],
                 [4., 5., 0., 0.],
                 [0., 0., 0., 0.]],
                [[3., 4., 0., 0.],
                 [7., 8., 9., 6.],
                 [1., 2., 3., 0.]]])
    """
    # Adopted from: https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699
    words = max_sent_length if max_sent_length else max([len(row) for batch in tokens for row in batch])
    sentences = max_sents if max_sents else max([len(batch) for batch in tokens])
    padded = [
        batch + [[0] * words] * (sentences - len(batch)) if pad_in_the_end
        else batch + (sentences - len(batch)) * [[0] * words]
        for batch in tokens
    ]
    padded = torch.Tensor([
        row + [0] * (words - len(row)) if pad_in_the_end
        else [0] * (words - len(row)) + row
        for batch in padded for row in batch
    ])
    padded = padded.view(-1, sentences, words)
    return padded


def custom_collate(x):
    items, distances, geo, target, valid_elem = zip(*x)

    items = pad_sequence([torch.Tensor(t[::-1]) for t in items], batch_first=True).flip(1)
    distances = pad_sequence([torch.Tensor(t[::-1]) for t in distances], batch_first=True).flip(1)

    geo = pad_sequence([torch.Tensor(t[::-1]) for t in geo], batch_first=True).flip(1)
    target = pad_sequence([torch.Tensor(t[::-1]) for t in target], batch_first=True).flip(1)
    valid_elem = torch.Tensor([t for t in valid_elem])

    return {
        'items': items,
        'distances': distances,
        'geo': geo,
        'target': target,
        'valid_elem': valid_elem
    }
