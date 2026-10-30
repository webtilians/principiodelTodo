"""One-time unification audit: preserve source blobs and all original PR tips."""
import json
import subprocess
from pathlib import Path

def git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()

def main():
    data=json.loads(Path("docs/repository-snapshot-2026-10-30.json").read_text())
    for source in data["sources"]:
        subprocess.run(["git","merge-base","--is-ancestor",source["sha"],"HEAD"],check=True)
    for path,expected in data["preserved_files"].items():
        current=data["relocated_verbatim"].get(path,path)
        actual=git("rev-parse","HEAD:"+current)
        if actual!=expected:
            raise AssertionError(f"Source content changed: {path} -> {current}")
    print(f"PRESERVATION_OK: {len(data['sources'])} PR tips and {len(data['preserved_files'])} exact source files")
if __name__=="__main__":
    main()
