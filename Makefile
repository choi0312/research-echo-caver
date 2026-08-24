.PHONY: verify test figure paper

verify:
	python3 scripts/verify_archive.py

test:
	python3 -m unittest discover -s tests -v

figure:
	python3 scripts/build_architecture_figure.py

paper: figure
	python3 scripts/build_paper_pdf.py
