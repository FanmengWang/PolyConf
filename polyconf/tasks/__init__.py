# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import importlib

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("polyconf.tasks." + file.name[:-3])
