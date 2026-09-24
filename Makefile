# Thin aliases for scripts/. `make help` lists targets.
.PHONY: help setup setup-gpu quickcheck test data p0 p1-cpu p1-gpu p2 p3 baselines p4 p5 battery controls topology verify clean

help:
	@echo "setup | setup-gpu | quickcheck | test"
	@echo "data | p0 | p1-cpu | p1-gpu | p2 | p3 | baselines | p4 | p5"
	@echo "battery | controls | topology | verify | clean"

setup:      ; bash scripts/00_setup.sh
setup-gpu:  ; bash scripts/00_setup.sh --gpu
quickcheck: ; bash scripts/01_quickcheck.sh
test:       ; .venv/bin/python -m pytest -q tests
data:       ; bash scripts/02_download_data.sh
p0:         ; bash scripts/03_build_p0.sh
p1-cpu:     ; bash scripts/04_build_p1_cpu.sh
p1-gpu:     ; bash scripts/05_build_p1_gpu.sh
p2:         ; bash scripts/06_run_p2_reliability.sh
p3:         ; bash scripts/07_run_p3_detection.sh
baselines:  ; bash scripts/08_run_baselines.sh
p4:         ; bash scripts/09_run_p4_robustness.sh
p5:         ; bash scripts/10_run_p5_transfer.sh
battery:    ; bash scripts/11_run_reliability_battery.sh
controls:   ; bash scripts/12_run_detection_controls.sh
topology:   ; bash scripts/13_run_topology_pilot.sh
verify:     ; bash scripts/14_figures_and_verify.sh
clean:      ; rm -rf outputs .pytest_cache; find . -name __pycache__ -prune -exec rm -rf {} +
