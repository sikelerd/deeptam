from abc import ABC, abstractmethod


class TrackingNetworkBase(ABC):

    def __init__(self):
        self._placeholders = {}

    @property
    def placeholders(self):
        """All placeholders required for feeding this network"""
        return self._placeholders

    @abstractmethod
    def build_net(self, image_key, point_key, image_current, intrinsics, prev_rotation, prev_translation):
        """Build the tracking network

        image_key: the image of the key frame
        image_current: the current image
        intrinsics: the camera intrinsics
        prev_rotation: the current guess for the camera rotation as angle axis representation
        prev_translation: the current guess for the camera translation

        Returns all network outputs as a dict.
        The following must be returned:

            predict_rotation
            predict_translation

        """
        pass

    @abstractmethod
    def build_training_net(self, image_key, point_key, image_current, point_current, intrinsics, prev_rotation, prev_translation, gt_rotation, gt_translation):
        pass
