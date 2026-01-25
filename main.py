from utils.helpers import base64_to_pil
from perception.visual import VisualPerception

if __name__ == "__main__":
    model = VisualPerception()
    img = "database/Gemini_Generated_Image_e2nbnue2nbnue2nb.png"
    model.generate_caption(img)