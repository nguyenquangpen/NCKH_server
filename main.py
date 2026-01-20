from utils.helpers import base64_to_pil
from perception.visual import VisualPerception

if __name__ == "__main__":
    model = VisualPerception()
    frame_b64 = "iVBORw0KGgoAAAANSUhEUgAAAuQAAAHtCAYAAABRQ2swAAAQAElEQVR4Acz9CbhtSXbXB/5XxN5nuvO7b8o35Mt5zppVg2rQgDCi"
    image_pil = base64_to_pil(frame_b64)
    model.generate_caption(image_pil)