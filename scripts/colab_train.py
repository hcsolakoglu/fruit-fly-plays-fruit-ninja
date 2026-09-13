import tarfile

with tarfile.open("/content/fruit-code.tar.gz") as t:
    t.extractall("/content/fruit-fly", filter="data")
import sys

for k in list(sys.modules):
    if k.startswith("flyfruit"):
        del sys.modules[k]
exec(
    compile(
        open("/content/fruit-fly/scripts/train_readout.py").read(),
        "/content/fruit-fly/scripts/train_readout.py",
        "exec",
    )
)
