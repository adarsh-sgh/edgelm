PY ?= .venv/bin/python
JOBS ?= 8
M = models/smollm2-135m
export EDGELM_TESTDATA := $(CURDIR)/testdata

.PHONY: build test asan tsan tiny venv model check bench eval clean

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j $(JOBS)

venv:
	python3 -m venv .venv && $(PY) -m pip install -q -r tools/requirements.txt

tiny:
	$(PY) tools/make_tiny.py --out testdata

test: build tiny
	./build/edgelm_tests

asan: tiny
	cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DEDGELM_SANITIZE=ON >/dev/null && cmake --build build-asan -j $(JOBS)
	./build-asan/edgelm_tests

tsan: tiny
	cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-fsanitize=thread \
	  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread >/dev/null && cmake --build build-tsan -j $(JOBS)
	./build-tsan/edgelm_tests

model:  ## download SmolLM2-135M and export f32 / q8 / q4 / q4all
	PY=$(PY) tools/download.sh

check: build  ## parity vs NumPy + HF transformers (needs torch, transformers, tokenizers in the venv)
	$(PY) tools/check_real.py --hf-dir $(M) --models $(M).f32.elm $(M).q8.elm $(M).q4.elm $(M).q4all.elm

bench: build
	tools/bench.sh

eval: build
	for d in q8 q4 q4all; do for a in f32 int8; do ./build/edgelm eval -m $(M).$$d.elm --ref $(M).f32.elm --act $$a --threads 4; done; done

clean:
	rm -rf build build-asan build-tsan testdata
