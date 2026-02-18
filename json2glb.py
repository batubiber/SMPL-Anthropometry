import json, base64
data = json.load(open('response.json'))
glb = base64.b64decode(data['model_glb'])
open('body_model.glb', 'wb').write(glb)
print(f'Saved {len(glb):,} bytes')