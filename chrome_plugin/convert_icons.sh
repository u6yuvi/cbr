#!/bin/bash

# Create images directory if it doesn't exist
mkdir -p images

# Convert SVG to different PNG sizes
convert -background none -size 16x16 images/icon.svg images/icon16.png
convert -background none -size 48x48 images/icon.svg images/icon48.png
convert -background none -size 128x128 images/icon.svg images/icon128.png 