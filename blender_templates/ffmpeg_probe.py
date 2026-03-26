import bpy
from pathlib import Path

print('enum:', [i.identifier for i in bpy.types.ImageFormatSettings.bl_rna.properties['file_format'].enum_items])
scene = bpy.data.scenes.get('MASTER_SEQUENCE')
if scene is None:
    scene = bpy.context.scene
print('scene name:', scene.name)
print('before:', scene.render.image_settings.file_format)
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
print('after:', scene.render.image_settings.file_format)
