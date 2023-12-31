SHELL := /bin/bash

clone_repo:
	git clone https://huggingface.co/microsoft/git-base-textcaps && cd git-base-textcaps && git lfs pull

qa:
	isort src/ tests/ && flake8 src/ tests/

clean:
	rm -rf out/

out:
	mkdir out

run_local:
	PYTHONPATH=src python src/image_captions/best/app.py

test_server:
	curl -w '\ntotal request time=%{time_total}\n' --data-binary @tests/data/cats.jpg http://0.0.0.0:8888

build_docker:
	docker build -t freepik/image_captions .

run_docker:
	docker run -p 8888:8888 --name image_captions --gpus all freepik/image_captions

stop_container:
	docker stop image_captions

rm_container:
	docker rm image_captions

benchmark: out tests/data/cats.jpg
	for c in {1,8,16,32,64}; do \
		ab -n 256 -c $${c} \
			-g out/requests_$${c}.dat \
			-p tests/data/cats.jpg \
			"http://0.0.0.0:8888/"; \
		cp out/requests_baseline_$${c}.dat out/requests_baseline.dat; \
		cp out/requests_$${c}.dat out/requests.dat; \
		gnuplot plot_response_time.p; \
		mv response_time.png "out/response_time_c$${c}.png"; \
	done

baseline: out tests/data/cats.jpg
	for c in {1,8,16,32,64}; do \
		ab -n 256 -c $${c} \
			-g out/requests_baseline_$${c}.dat \
			-p tests/data/cats.jpg \
			"http://0.0.0.0:8888/"; \
	done


images: out tests/data/cats.jpg
	for c in {1,8,16,32,64}; do \
		cp out/requests_baseline_$${c}.dat out/requests_baseline.dat; \
		cp out/requests_$${c}.dat out/requests.dat; \
		gnuplot plot_response_time.p; \
		mv response_time.png "out/response_time_c$${c}.png"; \
	done

batch_performance:
	PYTHONPATH=src python scripts/batch_performance.py