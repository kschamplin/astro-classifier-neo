# New astro classifier repo

This repository contains the code + notebooks for the development of novel
classifiers of astrophysical transients


## Goals
- Abstract data input pipeline for multiple dataset support
- Develop novel models for experimentation
- Explore active learning or RL learning

## Running the project

The project dependencies are managed with `poetry`. This has the advantage of
lockfiles, which makes environments reproducible. To install:
```bash
pyenv install 3.9.1
pyenv local 3.9.1
poetry install -vv
poetry run python -m ipykernel install --user --name pytorch
jupyter lab
```

## Development
CI will automatically run tests + type checks + linting on the project on every
commit. On a tagged push, it will create a release and push it to the package
registry. 
