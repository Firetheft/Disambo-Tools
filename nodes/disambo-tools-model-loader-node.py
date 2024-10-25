import glob
import os
from nodes import LoraLoader, CheckpointLoaderSimple
import folder_paths
from server import PromptServer
from folder_paths import get_directory_by_type
from aiohttp import web
import shutil


@PromptServer.instance.routes.get("/pysssss/view/{name}")
async def view(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    image_path = folder_paths.get_full_path(
        type, name)
    if not image_path:
        return web.Response(status=404)

    filename = os.path.basename(image_path)
    return web.FileResponse(image_path, headers={"Content-Disposition": f"filename=\"{filename}\""})


@PromptServer.instance.routes.post("/pysssss/save/{name}")
async def save_preview(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    body = await request.json()

    dir = get_directory_by_type(body.get("type", "output"))
    subfolder = body.get("subfolder", "")
    full_output_folder = os.path.join(dir, os.path.normpath(subfolder))

    if os.path.commonpath((dir, os.path.abspath(full_output_folder))) != dir:
        return web.Response(status=400)

    filepath = os.path.join(full_output_folder, body.get("filename", ""))
    image_path = folder_paths.get_full_path(type, name)
    image_path = os.path.splitext(
        image_path)[0] + os.path.splitext(filepath)[1]

    shutil.copyfile(filepath, image_path)

    return web.json_response({
        "image":  type + "/" + os.path.basename(image_path)
    })


@PromptServer.instance.routes.get("/pysssss/examples/{name}")
async def get_examples(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    file_path = folder_paths.get_full_path(
        type, name)
    if not file_path:
        return web.Response(status=404)
    
    file_path_no_ext = os.path.splitext(file_path)[0]
    examples = []

    if os.path.isdir(file_path_no_ext):
        examples += sorted(map(lambda t: os.path.relpath(t, file_path_no_ext),
                        glob.glob(file_path_no_ext + "/*.txt")))
        
    if os.path.isfile(file_path_no_ext + ".txt"):
        examples += ["notes"]
   
    return web.json_response(examples)

@PromptServer.instance.routes.post("/pysssss/examples/{name}")
async def save_example(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]
    body = await request.json()
    example_name = body["name"]
    example = body["example"]

    file_path = folder_paths.get_full_path(
        type, name)
    if not file_path:
        return web.Response(status=404)
    
    if not example_name.endswith(".txt"):
        example_name += ".txt"

    file_path_no_ext = os.path.splitext(file_path)[0]
    example_file = os.path.join(file_path_no_ext, example_name)
    if not os.path.exists(file_path_no_ext):
        os.mkdir(file_path_no_ext)
    with open(example_file, 'w', encoding='utf8') as f:
        f.write(example)

    return web.Response(status=201)


def populate_items(names, type):
    for idx, item_name in enumerate(names):

        file_name = os.path.splitext(item_name)[0]
        file_path = folder_paths.get_full_path(type, item_name)

        if file_path is None:
            print(f"(pysssss:better_combos) Unable to get path for {type} {item_name}")
            continue

        file_path_no_ext = os.path.splitext(file_path)[0]

        for ext in ["png", "jpg", "jpeg", "preview.png", "preview.jpeg"]:
            has_image = os.path.isfile(file_path_no_ext + "." + ext)
            if has_image:
                item_image = f"{file_name}.{ext}"
                break

        names[idx] = {
            "content": item_name,
            "image": f"{type}/{item_image}" if has_image else None,
        }
    names.sort(key=lambda i: i["content"].lower())


class LoraLoaderNode(LoraLoader):
    RETURN_TYPES = (*LoraLoader.RETURN_TYPES, "STRING",)
    FUNCTION = "load_lora"
    CATEGORY = "📜Disambo Tools"

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        names = types["required"]["lora_name"][0]
        populate_items(names, "loras")
        types["optional"] = { "prompt": ("HIDDEN",) }
        return types

    @classmethod
    def VALIDATE_INPUTS(s, lora_name):
        types = super().INPUT_TYPES()
        names = types["required"]["lora_name"][0]

        name = lora_name["content"]
        if name in names:
            return True
        else:
            return f"Lora not found: {name}"

    def load_lora(self, **kwargs):
        print("kwargs before super call:", kwargs)  # Add this line for debugging
        kwargs["lora_name"] = kwargs["lora_name"]["content"]
        prompt = kwargs.pop("prompt", "")
        print("lora_name:", kwargs["lora_name"], type(kwargs["lora_name"])) # Add this to check type
        return (*super().load_lora(**kwargs), prompt)


class CheckpointLoaderNode(CheckpointLoaderSimple):
    RETURN_TYPES = (*CheckpointLoaderSimple.RETURN_TYPES, "STRING",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "📜Disambo Tools"

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        names = types["required"]["ckpt_name"][0]
        populate_items(names, "checkpoints")
        types["optional"] = { "prompt": ("HIDDEN",) }
        return types

    @classmethod
    def VALIDATE_INPUTS(s, ckpt_name):
        types = super().INPUT_TYPES()
        names = types["required"]["ckpt_name"][0]

        name = ckpt_name["content"]
        if name in names:
            return True
        else:
            return f"Checkpoint not found: {name}"

    def load_checkpoint(self, **kwargs):
        kwargs["ckpt_name"] = kwargs["ckpt_name"]["content"]
        prompt = kwargs.pop("prompt", "")
        return (*super().load_checkpoint(**kwargs), prompt)


class MultiLoraLoaderNode:
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_multiple_loras"
    CATEGORY = "📜Disambo Tools"

    @classmethod
    def INPUT_TYPES(cls):
        lora_names = folder_paths.get_filename_list("loras")
        populate_items(lora_names, "loras") #Use existing populate_items
        lora_options = ({"content": "None", "image": None}, *lora_names)

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora1": (lora_options, {}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora2": (lora_options, {}),
                "strength2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora3": (lora_options, {}),
                "strength3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora4": (lora_options, {}),
                "strength4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {},
        }

    def load_multiple_loras(self, model, clip, lora1, strength1, lora2, strength2, lora3, strength3, lora4, strength4, **kwargs):
        lora_loader = LoraLoaderNode()
        loras_and_strengths = [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
        ]

        for lora, strength in loras_and_strengths:
            if lora["content"] != "None" and strength != 0:
                try:
                    model, clip, *rest = lora_loader.load_lora(model=model, clip=clip, lora_name={"content": lora["content"], "image": lora["image"]}, strength_model=strength, strength_clip=strength)
                except Exception as e:
                    print(f"Error loading Lora {lora['content']}: {e}")
                    return (model, clip)

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderNode": LoraLoaderNode,
    "CheckpointLoaderNode": CheckpointLoaderNode,
    "MultiLoraLoaderNode": MultiLoraLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderNode": "Lora Loader Node",
    "CheckpointLoaderNode": "Checkpoint Loader Node",
    "MultiLoraLoaderNode": "Multi Lora Loader Node",
}
