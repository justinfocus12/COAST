#!/bin/bash
fn=$1
pandoc "$fn".md -o "$fn".pdf --pdf-engine=pdflatex -V geometry:margin=1in --citeproc
