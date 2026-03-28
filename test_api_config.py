from src.common.config import load_config

cfg = load_config()
print("Groq API Key:", "SET" if cfg.groq_api_key else "NOT SET")
print("Groq Base URL:", cfg.groq_base_url)
print("Groq Model:", cfg.groq_model)
print("NVIDIA API Key:", "SET" if cfg.nvidia_api_key else "NOT SET")
print("NVIDIA Chat:", cfg.nvidia_api_key and "Enabled" or "Disabled")
print("Clipdrop API Key:", "SET" if cfg.clip_drop_api_key else "NOT SET")
print("SerpApi Key:", "SET" if cfg.serpapi_api_key else "NOT SET")
print("Pexels API Key:", "SET" if cfg.pexels_api_key else "NOT SET")
