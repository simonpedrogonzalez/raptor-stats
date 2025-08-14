.PHONY: test prof memprof compare cleanexp style docfix docstyle doc readme pubdoc build upload upload-test

test:
	pytest tests --maxfail=1 --pdb

prof:
	kernprof -l  -o experiments/profile_method.py.lprof experiments/profile_method.py
	python3 -m line_profiler experiments/profile_method.py.lprof > experiments/profile_results.txt

memprof:
	python3 -m memory_profiler experiments/profile_method.py > experiments/memory_results.txt

compare:
	python3 experiments/compare_vs_rasterstats.py

cleanexp:
	rm -rf results/*

style:
	isort --profile black raptorstats/*.py tests/*.py
	black --line-length 88 --preview raptorstats/*.py tests/*.py

docfix:
	docformatter -i --wrap-descriptions 88 --blank --wrap-summaries 88 -r raptorstats/

docstyle:
	pydocstyle raptorstats/ --convention=numpy --add-ignore=D100,D

doc:
	cd docs && make clean
	cd docs && sphinx-apidoc -o . ../raptorstats -f -e
	cd docs && make html

readme:
	python -c "from m2r import parse_from_file; \
		readme = parse_from_file('README.md'); \
		open('docs/readme.rst', 'w').write(readme)"

pubdoc:
	rsync -av --delete --exclude '.git/' docs/_build/html/ ../raptor-stats-docs/
	touch ../raptor-stats-docs/.nojekyll
	cd ../raptor-stats-docs && \
		git checkout -B master && \
		git add -A && \
		git commit -m "update docs" || true && \
		git push -f origin master

build:
	@rm -rf build dist *.egg-info
	python -m build --sdist --wheel

upload:
	@echo "Uploading to PyPI..."
	python -m twine check dist/*
	python -m twine upload dist/* --verbose

upload-test:
	@echo "Uploading to Test PyPI..."
	python -m twine check dist/*
	python -m twine upload --repository testpypi dist/* --verbose

