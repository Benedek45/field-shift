#!/bin/sh
# Build bench/paper/b0_v0.5.pdf from the Markdown source (pandoc + WeasyPrint; no LaTeX needed).
set -eu
cd "$(dirname "$0")"
pandoc b0_v0.5.md --standalone --css paper.css --resource-path=. \
    --pdf-engine=weasyprint -o b0_v0.5.pdf
