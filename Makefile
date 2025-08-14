.PHONY: test

test:
	pytest tests --maxfail=1 --pdb

prof:
	kernprof -l  -o experiments/profile_method.py.lprof experiments/profile_method.py
	python3 -m line_profiler experiments/profile_method.py.lprof > experiments/profile_results.txt

memprof:
	python3 -m memory_profiler experiments/profile_method.py > experiments/memory_results.txt

compare:
	uv run experiments/compare_vs_rasterstats.py

cleanexp:
	rm -rf results/*

style:
	isort --profile black raptorstats/*.py tests/*.py
	black --line-length 88 --preview raptorstats/*.py tests/*.py