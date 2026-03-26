import bpy
scene = bpy.context.scene
render_props = scene.render.bl_rna.properties.keys()
print('has render.file_format:', 'file_format' in render_props)
if 'file_format' in render_props:
    prop = scene.render.bl_rna.properties['file_format']
    print('render.file_format enums:', [e.identifier for e in prop.enum_items])
    print('before render.file_format:', scene.render.file_format)
    scene.render.file_format = 'FFMPEG'
    print('after render.file_format:', scene.render.file_format)
else:
    print('render properties sample:', list(render_props)[:30])
