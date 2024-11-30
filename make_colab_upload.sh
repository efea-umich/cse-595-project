#!/bin/bash

export COPYFILE_DISABLE=true
export GZIP=-9
tar cvfz upload.tar.gz src/ *.yaml pyproject.toml uv.lock data/