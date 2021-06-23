# New astro classifier repo
[![coverage report](https://gitlab.com/saji.champlin/astro-classifier-neo/badges/master/coverage.svg)](https://gitlab.com/saji.champlin/astro-classifier-neo/-/commits/master)
[![pipeline status](https://gitlab.com/saji.champlin/astro-classifier-neo/badges/master/pipeline.svg)](https://gitlab.com/saji.champlin/astro-classifier-neo/-/commits/master)

This repository contains the code + notebooks for the development of novel
classifiers of astrophysical transients


## Goals
- Abstract data input pipeline for multiple dataset support
- Develop novel models for experimentation
- Explore active learning or RL learning

## Running the project

The project has a provided conda environment that contains all the dependencies needed for working
with the repository. 

## Development
CI will automatically run tests + type checks + linting on the project on every
commit. On a tagged push, it will create a release and push it to the package
registry.
