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
                await websocket.send_text("unloaded_florence")
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
                    prompt_json_path = data.get("prompt_json_path")
                    base_llama_dir = "llama_emb/tvsum_sum/" 
                    
                    print(f"🧠 Llama-3 is processing: {prompt_json_path}")
                    try:
                        # tool chua setup nhan error or failure
                        result = await asyncio.to_thread(
                            llama_view.process_metadata_to_h5, 
                            prompt_json_path, 
                            base_llama_dir
                        )
                        if result is True:
                            await websocket.send_json({
                                "status": "completed_llama",
                                "message": "Features extracted and saved successfully",
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
                print(f"Error processing message: {str(e)}")
                continue
    except Exception as e:
        print(f"Connection closed: {str(e)}")
    finally:
        async with model_lock:
            await asyncio.to_thread(florence_view.unload_model)
            await asyncio.to_thread(llama_view.unload_model)
        print("Models unloaded, connection closed.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



