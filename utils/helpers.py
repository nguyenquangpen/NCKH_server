import base64
from io import BytesIO
from PIL import Image

def base64_to_pil(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None