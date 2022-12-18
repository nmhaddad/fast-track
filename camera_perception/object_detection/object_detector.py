
class ObjectDetector:

    def __init__(self, image_shape):

        self.detector = None
        self.image_shape = image_shape

    @property
    def image_width(self) -> int:
        return self.image_shape[0]
    
    @property
    def image_height(self)-> int:
        return self.image_shape[1]

    def detect(self, image):
        return image
