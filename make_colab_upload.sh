#!/bin/bash

export COPYFILE_DISABLE=true
# make tar
tar cvfz upload.tar.gz src/ *.yaml pyproject.toml uv.lock