import uuid


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

    def gen_uuid():
        """
        Returns a unique global annotation id. UUID instead of incremental ids
        """
        return uuid.uuid4()
