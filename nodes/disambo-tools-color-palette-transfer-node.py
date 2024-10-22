import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import re
import ast


def EuclideanDistance(detected_colors, target_colors):
    return np.linalg.norm(detected_colors - target_colors, axis=1)


def ManhattanDistance(detected_colors, target_colors):
    return np.sum(np.abs(detected_colors - target_colors), axis=1)


def ColorClustering(image, k, cluster_method):
    img_array = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster_methods = {
        "Kmeans": KMeans,
        "Mini batch Kmeans": MiniBatchKMeans
    }

    clustering_model = cluster_methods.get(cluster_method)(n_clusters=k, n_init='auto')

    clustering_model.fit(img_array)
    main_colors = clustering_model.cluster_centers_
    return image, main_colors.astype(int), clustering_model


def SwitchColors(image, detected_colors, target_colors, clustering_model, distance_method):
    closest_colors = []

    distance_methods = {
        "Euclidean": EuclideanDistance,
        "Manhattan": ManhattanDistance
    }

    distance_method = distance_methods.get(distance_method)

    for color in detected_colors:
        distances = distance_method(color, target_colors)
        closest_color = target_colors[np.argmin(distances)]
        closest_colors.append(closest_color)

    closest_colors = np.array(closest_colors)

    image = closest_colors[clustering_model.labels_].reshape(image.shape)
    image = np.array(image).astype(np.float32) / 255.0
    processedImage = torch.from_numpy(image)[None,]

    return processedImage


def Hex_to_RGB(inhex: str) -> tuple:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code: {inhex}')
    else:
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)


def parse_color_palette(color_palette: str):
    # Check if the input contains hex codes (starts with #) or RGB tuples
    hex_pattern = r'#([A-Fa-f0-9]{6})'
    rgb_pattern = r'\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)'

    if re.search(hex_pattern, color_palette):
        # Input is in hex format, split by commas and convert to RGB
        hex_colors = [color.strip() for color in color_palette.split(',')]
        return [Hex_to_RGB(color) for color in hex_colors]

    elif re.search(rgb_pattern, color_palette):
        # Input is in RGB tuple format, use ast.literal_eval to safely convert it to a list of tuples
        rgb_colors_str = f"[{color_palette}]"  # Add brackets to make it a list of tuples
        return ast.literal_eval(rgb_colors_str)

    else:
        raise ValueError("Invalid color palette format. Please use hex format '#RRGGBB' or RGB format '(R, G, B)'.")


class ColorPaletteTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": ("IMAGE",),
                "color_palette": ("STRING", {"forceInput": True}),  # 使用 forceInput 强制从外部接收输入
                # "color_palette": ("STRING", {'default': '', 'multiline': True}),
                "cluster_method": (["Kmeans", "Mini batch Kmeans"], {'default': 'Kmeans'}),
                "distance_method": (["Euclidean", "Manhattan"], {'default': 'Euclidean'}),
            }
        }
        return data_in

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"
    CATEGORY = "📜Disambo Tools"

    def color_transfer(self, image, color_palette, cluster_method, distance_method):
        # Parse color_palette string into a list of RGB tuples
        target_colors = parse_color_palette(color_palette)

        if len(target_colors) == 0:
            return (image,)

        processedImages = []

        for img in image:
            img = 255. * img.cpu().numpy()

            clustered_img, detected_colors, clustering_model = ColorClustering(img, len(target_colors), cluster_method)
            processed = SwitchColors(clustered_img, detected_colors, target_colors, clustering_model, distance_method)
            processedImages.append(processed)

        output = torch.cat(processedImages, dim=0)

        return (output, )

# Node export details
NODE_CLASS_MAPPINGS = {
    "ColorPaletteTransferNode": ColorPaletteTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteTransferNode": "Color Palette Transfer Node"
}