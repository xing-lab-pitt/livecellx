class AnnotationIdGenerator:
    """
    Class for generating unique annotation ids.
    """

    global_id = 0

    def __init__(self):
        self._id = 0

    def gen_id(self):
        """
        Returns a unique annotation id.
        """
        self._id += 1
        return self._id

    def gen_global_id(self):
        """
        Returns a unique global annotation id.
        """
        AnnotationIdGenerator.global_id += 1
        return AnnotationIdGenerator.global_id
