.PHONY: verify test paper

verify:
	python3 scripts/verify_archive.py

test:
	python3 -m unittest discover -s tests -v

paper:
	python3 scripts/build_paper_pdf.py

