import mahotas.features.texture


def get_texture_features(image, **kwargs):
    """Returns a list of texture features for the given image.

    Parameters
    ----------
    image : ndarray
        The image to extract features from.

    Returns
    -------
    list
        A list of texture features.
    """
    return mahotas.features.texture.haralick(image, **kwargs)
