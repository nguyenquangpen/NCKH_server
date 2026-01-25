from fastapi import FastAPI, WebSocket, APIRouter
import uvicorn
import base64
import json
from utils.helpers import *
from Llama_3.view import LlamaView
from Florence.view import FlorenceView

app = FastAPI()
router = APIRouter()
florence_view = FlorenceView()
llama_view = LlamaView()

@app.websocket("/ws/agent")
async def video_handler(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            msg = await websocket.receive_json()
            if msg == "init_florence":
                # setup florence model here
                await websocket.send_text("ready_florence")
            
            elif msg == "init_llama":
                # setup llama model here
                await websocket.send_text("ready_llama")
            
            elif msg == "success_florence":
                # setup clear vram here
                pass
            
            elif msg == "success_llama":
                # setup clear vram here
                pass
            
            try:
                data = json.loads(msg)
                if data.get("Status") == "run":
                    shot_id = data.get("shot_id")
                    frame_b64 = data.get("frame")
                    print(f"Processing shot: {shot_id}")

                    image_pil = base64_to_pil(frame_b64)
                    caption = florence_view.generate_caption(image_pil)
                    result = {
                        "shot_id": shot_id,
                        "visual_description": caption,
                        "status": "completed"
                    }
                    await websocket.send_json(result)
                elif data.get("Status") == "run_llama":
                    pass
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                continue
    except Exception as e:
        print(f"Connection closed: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



