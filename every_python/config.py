import dataclasses
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path.home() / ".every-python"
CONFIG_FILE = f"{BASE_DIR}/config.json"
CPYTHON_REPO = "https://github.com/python/cpython.git"


@dataclasses.dataclass
class Remote:
    name: str
    url: str
    repo_dir: str


class Config:
    def load(self) -> dict[str, Any]:
        try:
            with open(CONFIG_FILE, "r") as f:
                data = f.read()
            return json.loads(data)
        except FileNotFoundError:
            return {"remotes": []}

    def save(self, data: str):
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            f.write(data)
    
    def get_remote(self, name: str) -> Remote:
        config = self.load()
        if len(config.get("remotes", [])) == 0:
            return Remote(
                name="default",
                url=CPYTHON_REPO,
                repo_dir="cpython",
            )
        
        remote = next((r for r in config["remotes"] if r["name"] == name), None)
        if remote is None:
            raise ValueError(f"Remote '{name}' not found in configuration.")
        return Remote(
            name=remote["name"],
            url=remote["url"],
            repo_dir=remote["repo_dir"],
        )
        