import bpy
from pathlib import Path

out_path = Path(r"D:\ET Hackathon\blender_templates\base_templates_45.blend")
out_path.parent.mkdir(parents=True, exist_ok=True)

required = [
    "headline_card",
    "image_plus_text_split",
    "quote_card",
    "bullet_scene",
    "outro_card",
]

for name in required:
    scene = bpy.data.scenes.get(name)
    if scene is None:
        scene = bpy.data.scenes.new(name)
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.fps = 30

    cam = None
    for obj in scene.objects:
        if obj.type == 'CAMERA':
            cam = obj
            break
    if cam is None:
        cam_data = bpy.data.cameras.new(f"{name}_Camera")
        cam = bpy.data.objects.new(f"{name}_Camera", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (0.0, -8.0, 0.0)
        cam.rotation_euler = (1.5708, 0.0, 0.0)
    scene.camera = cam

    for obj in list(scene.objects):
        if obj.type == 'FONT' and obj.name.startswith(("TITLE_", "NARR_")):
            bpy.data.objects.remove(obj, do_unlink=True)

    title_curve = bpy.data.curves.new(type='FONT', name=f"{name}_TitleCurve")
    title_obj = bpy.data.objects.new(f"TITLE_{name}", title_curve)
    title_curve.body = "TITLE"
    title_curve.size = 0.7
    title_obj.location = (-4.5, 0.0, 2.7)
    scene.collection.objects.link(title_obj)

    narr_curve = bpy.data.curves.new(type='FONT', name=f"{name}_NarrCurve")
    narr_obj = bpy.data.objects.new(f"NARR_{name}", narr_curve)
    narr_curve.body = "NARRATION"
    narr_curve.size = 0.42
    narr_obj.location = (-4.5, 0.0, 1.7)
    scene.collection.objects.link(narr_obj)

bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
print(f"Saved template blend: {out_path}")
