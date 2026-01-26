import os
from fastapi import FastAPI, WebSocket, APIRouter
import uvicorn
import base64
import json
import asyncio
from Llama_3.view import LlamaView
from Florence.view import FlorenceView
from Florence.model import FlorenceInput, FlorenceOutput

app = FastAPI()
florence_view = FlorenceView()
llama_view = LlamaView()

model_lock = asyncio.Lock()
model_loaded = False

@app.websocket("/ws/agent")
async def video_handler(websocket: WebSocket):
    await websocket.accept()
    global model_loaded
    loop = asyncio.get_event_loop()
    print("Client connected")

    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "init_florence":
                async with model_lock:
                    if not model_loaded:
                        await loop.run_in_executor(None, florence_view._load_model)
                        model_loaded = True
                print("Florence-2 model is ready")
                await websocket.send_text("ready_florence")
                continue

            elif msg == "success_florence":
                async with model_lock:
                    if model_loaded:
                        await loop.run_in_executor(None, florence_view.unload_model)
                        model_loaded = False
                continue
            
            elif msg == "init_llama":
                # async with model_lock:
                #     if not model_loaded:
                #         await asyncio.to_thread(llama_view._load_model)
                #         model_loaded = True
                await websocket.send_text("ready_llama")
                continue
            
            elif msg == "success_llama":
                # setup clear vram here
                # llama_view.unload_model()
                continue
            
            try:
                print("run here")
                data = json.loads(msg)
                if data.get("status") == "run_florence":
                    shot_id = data.get("shot_id")
                    image_b64 = data.get("image_b64")
                    print(f"Processing shot: {shot_id}")

                    input_obj = FlorenceInput(
                        shot_id=shot_id,
                        image_b64=image_b64
                    )

                    print(f"🚀 Đang chạy Florence cho Shot {shot_id}...")
                    caption_result = await loop.run_in_executor(
                        None, 
                        florence_view.generate_caption, 
                        input_obj
                    )

                    print("Caption generated")
                    if caption_result:
                        response = {
                            "shot_id": shot_id,
                            "caption": caption_result,
                            "status": "completed"
                        }
                        await websocket.send_json(response)
                    else:
                        await websocket.send_json({"status": "error", "message": "Inference failed"})

                elif data.get("status") == "run_llama":
                    pass
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                continue
    except Exception as e:
        print(f"Connection closed: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



