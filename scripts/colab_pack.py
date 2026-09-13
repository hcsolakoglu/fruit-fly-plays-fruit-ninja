import tarfile
from pathlib import Path

p = Path("/content/fruit-fly")
with tarfile.open("/content/fruit-results.tar.gz", "w:gz") as t:
    for f in p.joinpath("data").iterdir():
        if f.suffix in (".npz", ".json"):
            t.add(f, arcname=str(f.relative_to(p)))
    for f in p.joinpath("results").glob("*.json"):
        t.add(f, arcname=str(f.relative_to(p)))
print("artifact_bytes", Path("/content/fruit-results.tar.gz").stat().st_size)
