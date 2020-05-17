build.document:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex

move.to.bin:
	mv *.pdf *.log ./bin

clean:
	[ -e *.aux ] && rm *.aux
	[ -e *.bbl ] && rm *.bbl
	[ -e *.blg ] && rm *.blg
	[ -e *.brf ] && rm *.brf
	[ -e *.idx ] && rm *.idx
	[ -e *.lof ] && rm *.lof
	[ -e *.lot ] && rm *.lot
	[ -e *.out ] && rm *.out
	[ -e *.toc ] && rm *.toc
	[ -e *.spl ] && rm *.spl
