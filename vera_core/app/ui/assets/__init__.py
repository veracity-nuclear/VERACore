import os

from trame.assets.local import LocalFileManager

base_dir = os.path.dirname(os.path.abspath(__file__))
asset_manager = LocalFileManager(base_dir)
asset_manager.url("logo", "./logo.svg")
LOGO = asset_manager.logo
