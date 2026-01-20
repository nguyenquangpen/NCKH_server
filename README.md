LLMVS_Agent/
├── main.py              # File chạy chính
├── bridge.py            # Module WebSocket
├── Model.py             # định nghĩa các Schema 
├── perception/          # Module
│   ├── visual.py        # Florence-2 logic
│   └── audio.py         # Whisper logic
├── reasoning/           # Module
│   ├── prompt_factory.py# Quản lý các mẫu Prompt
│   ├── SystemPrompt.md  # system prompt
│   └── llama_agent.py   # Llama-3 logic (Align & Score)
└── utils/               # Tiện ích (Convert base64, JSON parser...)