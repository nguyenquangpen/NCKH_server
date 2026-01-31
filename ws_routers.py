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

@app.websocket("/ws/agent")
async def video_handler(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    print("Client connected")

    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "init_florence":
                async with model_lock:
                    await asyncio.to_thread(florence_view._load_model)
                print("Florence-2 model is ready")
                await websocket.send_text("ready_florence")
                continue

            elif msg == "success_florence":
                async with model_lock:
                    print("🧹 Unloading Florence-2 to free VRAM...")
                    await asyncio.to_thread(florence_view.unload_model)
                # cần chỉnh bên client thằng này
                # await websocket.send_text("unloaded_florence")
                continue
            
            elif msg == "init_llama":
                async with model_lock:
                    await asyncio.to_thread(llama_view._load_model)
                await websocket.send_text("ready_llama")
                continue
            
            elif msg == "success_llama":
                async with model_lock:
                    print("🧹 Unloading Llama-3 to free VRAM...")
                    await asyncio.to_thread(llama_view.unload_model)
                # cần chỉnh bên client thằng này
                await websocket.send_text("unloaded_llama")
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
                    segment_data = data.get("segment_data")
                    print(f"🧠 Llama-3 is processing")
                    try:
                        # tool chua setup nhan error or failure
                        x1, x2 = await asyncio.to_thread(
                            llama_view.get_segment_embeddings,
                            segment_data,
                        )
                        if x1 is not None and x2 is not None:
                            await websocket.send_json({
                                "status": "completed_llama",
                                "x1": x1.tolist(),
                                "x2": x2.tolist(),
                            })
                        else:
                            await websocket.send_json({
                                "status": "failed_llama",
                                "message": "Llama-3 processing failed",
                            })
                    except Exception as e:
                        await websocket.send_json({
                            "status": "error_llama",
                            "message": f"Llama crashed: {str(e)}",
                        })
            except Exception as e:
                await websocket.send_json({
                    "status": "server_error",
                    "message": str(e)
                })
    except Exception as e:
        print(f"Connection closed: {str(e)}")
    finally:
        # async with model_lock:
        #     await asyncio.to_thread(florence_view.unload_model)
        #     await asyncio.to_thread(llama_view.unload_model)
        print("Models unloaded, connection closed.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



