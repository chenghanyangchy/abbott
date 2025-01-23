"""Generate JSON schemas for task arguments & write them to manifest."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "abbott"
    AUTHORS = "Ruth Hornbachner"
    docs_link = "https://github.com/pelkmanslab/abbott"
    if docs_link:
        create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)
    else:
        create_manifest(package=PACKAGE, authors=AUTHORS)
