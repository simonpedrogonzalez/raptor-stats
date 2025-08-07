.PHONY: test

test:
	pytest tests/test_zonal.py --maxfail=1 --pdb

prof:
	kernprof -l  -o experiments/profile_method.py.lprof experiments/profile_method.py
	python3 -m line_profiler experiments/profile_method.py.lprof > experiments/profile_results.txt
