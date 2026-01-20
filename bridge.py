from fastapi import FastAPI, WebSocket
import uvicorn
import base64
import json
from utils.helpers import *
from preception.visual import VisualPerceptionTool

app = FastAPI()

visual_tool = VisualPerceptionTool()

@app.websocket("/ws/agent")
async def video_handler(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_json()
            shot_id = data.get("shot_id")
            frame_b64 = data.get("frame")
            prompt = data.get("prompt")

            print(f"Processing shot: {shot_id}")

            image_pil = base64_to_pil(frame_b64)

            caption = visual_tool.generate_caption(image_pil, prompt)

            result = {
                "shot_id": shot_id,
                "caption": caption,
                "status": "completed"
            }
            await websocket.send_json(result)

    except Exception as e:
        print(f"Connection closed: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



