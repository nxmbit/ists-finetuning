.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	mkdir -p output
	cp build/main.pdf output/main.pdf

clean:
	latexmk -C -outdir=build main.tex
