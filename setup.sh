#!/bin/bash

# Fix pmdarima build issue by installing numpy first
pip install numpy==1.24.4

# Now install everything else
pip install -r requirements.txt

