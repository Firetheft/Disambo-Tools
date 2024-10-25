import importlib.util
import os
import importlib
import pkg_resources
import sys
import subprocess
import folder_paths


node_list = [
    "disambo-tools-flux-prompt-enhance-node",
    "disambo-tools-prompt-generator-node",
    "disambo-tools-gemini-flash-node",
    "disambo-tools-minicpm-v-2-6-int4-node",
    "disambo-tools-color-palette-extractor-node",
    "disambo-tools-color-palette-picker-node",
    "disambo-tools-color-palette-transfer-node",
    "disambo-tools-resharpen-details-node",
    "disambo-tools-model-loader-node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]