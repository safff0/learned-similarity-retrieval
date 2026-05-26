def collate_fn(items):
    """Combine a list of dataset items into a single batch dict.

    Expected shape: items[i] is a dict (e.g. {"data_object": tensor, "labels": int}).
    Return a dict where each key maps to a stacked/batched tensor.
    """
    raise NotImplementedError
