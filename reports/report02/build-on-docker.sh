# compile document using LaTeX in a docker container
#docker run --rm -v "$PWD":/home/latex -w /home/latex aergus/latex pdflatex --help
docker run --rm -v "$PWD":/home/latex -w /home/latex aergus/latex make build.document move.to.bin clean
