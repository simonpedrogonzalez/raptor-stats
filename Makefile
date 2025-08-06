.PHONY: test

test:
	pytest tests/test_zonal.py --maxfail=1 --pdb