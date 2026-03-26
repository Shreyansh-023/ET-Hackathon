import json
from pathlib import Path
from src.common.models import Article, ArticleUnderstanding, Scene
from src.common.validation import validate_payload
from src.planner.engine import (
    _article_understanding_from_text,
    _fallback_template_plan,
    _enforce_timing_policy,
    _finalize_scene_rows,
    _semantic_validate_scene_rows,
    _validate_scenes,
)

job_id = "20260324-064615-b73827a5"
job_dir = Path("jobs") / job_id

# Load article
article_json = json.loads((job_dir / "parsed" / "article.json").read_text())
article = validate_payload(Article, article_json)

# Generate understanding from text
understanding = _article_understanding_from_text(article)

# Generate fallback storyboard
scenes = _fallback_template_plan(article, understanding)

print(f"Generated {len(scenes)} scenes")
for scene in scenes:
    print(f"  {scene.scene_id}: {scene.type} ({scene.start:.1f}s-{scene.end:.1f}s) {scene.narration[:50]}")

# Write outputs
storyboard_dir = job_dir / "storyboard"
storyboard_dir.mkdir(parents=True, exist_ok=True)

with open(storyboard_dir / "article_understanding.json", "w") as f:
    json.dump(understanding.model_dump(), f, indent=2, sort_keys=True)

with open(storyboard_dir / "scenes.json", "w") as f:
    json.dump([s.model_dump() for s in scenes], f, indent=2, sort_keys=True)

with open(storyboard_dir / "storyboard.json", "w") as f:
    json.dump({"scenes": [s.model_dump() for s in scenes]}, f, indent=2, sort_keys=True)

# Mark as fallback
with open(storyboard_dir / "planner_prompt.txt", "w") as f:
    f.write("FALLBACK PLANNER\nLLM planner timed out, using deterministic template fallback.\n")

with open(storyboard_dir / "planner_output_raw.txt", "w") as f:
    f.write(json.dumps([s.model_dump() for s in scenes], indent=2))

print("\nFallback storyboard written to job directory")
