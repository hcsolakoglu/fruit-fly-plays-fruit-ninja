import tarfile

with tarfile.open("/content/fruit-code.tar.gz") as t:
    t.extractall("/content/fruit-fly", filter="data")
