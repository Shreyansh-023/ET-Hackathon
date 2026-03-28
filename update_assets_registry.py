import json
from pathlib import Path

job_id = "20260324-064615-b73827a5"
job_dir = Path("jobs") / job_id

# Read the images registry (created by visual planning)
with open(job_dir / "assets" / "visual_plan.json") as f:
    visual_plan = json.load(f)

# Build assets registry from images
images_dir = job_dir / "assets" / "images"
registry_items = []
scene_count = 0

for scene_id in [f"scene-{i:03d}" for i in range(1, 7)]:
    image_path = images_dir / f"{scene_id}.jpg"
    if image_path.exists():
        registry_items.append({
            "asset_id": f"asset-{scene_id}",
            "llm_plan": visual_plan["scene_plan"].get(scene_id, {}),
            "path": f"assets/images/{scene_id}.jpg",
            "provenance": {"provider": "serpapi", "status": "resolved"},
            "scene_id": scene_id,
            "source": "serpapi",
        })
        scene_count += 1

stock_queries = visual_plan.get("stock_queries", visual_plan.get("pexels_queries", []))

# Write assets registry
registry = {
    "assets": registry_items,
    "stock_query_count": len(stock_queries),
    "stock_queries": stock_queries,
    # Keep legacy aliases for older tooling.
    "pexels_query_count": len(stock_queries),
    "pexels_queries": stock_queries,
    "pool_size": 5,
    "scene_count": scene_count,
    "subtitles": {
        "srt": "audio/subtitles.srt",
        "vtt": "audio/subtitles.vtt"
    },
    "visual_plan": {
        "notes": visual_plan.get("notes"),
        "stock_query_count": len(stock_queries),
        # Keep legacy alias.
        "pexels_query_count": len(stock_queries),
    },
    "voiceover": {
        "duration_seconds": 77.72,
        "path": "audio/voiceover.wav"
    }}

with open(job_dir / "assets" / "assets_registry.json", "w") as f:
    json.dump(registry, f, indent=2, sort_keys=True)

print(f"Assets registry updated: {scene_count} images")
