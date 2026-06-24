APP_NAME ?= llm-infer-cal
IMAGE_REPOSITORY ?= 172.28.0.32:3443/xrwang/llm-infer-cal
IMAGE_TAG ?= latest
IMAGE ?= $(IMAGE_REPOSITORY):$(IMAGE_TAG)

HELM_CHART ?= charts/llm-infer-cal
HELM_RELEASE ?= llm-infer-cal
HELM_NAMESPACE ?= llm-infer-cal
HELM_PACKAGE_DIR ?= dist

DOCKER ?= docker
HELM ?= helm
CARGO ?= cargo
NPM ?= npm

.PHONY: help
help:
	@printf "Targets:\\n"
	@printf "  build          Build Rust workspace and frontend\\n"
	@printf "  test           Run Rust tests and frontend src tests\\n"
	@printf "  docker-build   Build Docker image ($(IMAGE))\\n"
	@printf "  docker-run     Run Docker image on localhost:8080\\n"
	@printf "  helm-lint      Lint Helm chart\\n"
	@printf "  helm-template  Render Helm chart\\n"
	@printf "  helm-package   Package Helm chart into $(HELM_PACKAGE_DIR)\\n"
	@printf "  helm-install   Install or upgrade Helm release\\n"
	@printf "  helm-uninstall Uninstall Helm release\\n"

.PHONY: build
build: frontend-build rust-build

.PHONY: rust-build
rust-build:
	$(CARGO) build --workspace

.PHONY: frontend-install
frontend-install:
	$(NPM) ci --prefix web/frontend

.PHONY: frontend-build
frontend-build: frontend-install
	$(NPM) run build --prefix web/frontend

.PHONY: test
test:
	$(CARGO) test --workspace
	$(NPM) test --prefix web/frontend -- src

.PHONY: docker-build
docker-build:
	$(DOCKER) build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	$(DOCKER) run --rm -p 8080:8080 $(IMAGE)

.PHONY: helm-lint
helm-lint:
	$(HELM) lint $(HELM_CHART)

.PHONY: helm-template
helm-template:
	$(HELM) template $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE) \
		--set-string image.repository=$(IMAGE_REPOSITORY) \
		--set-string image.tag=$(IMAGE_TAG)

.PHONY: helm-package
helm-package:
	mkdir -p $(HELM_PACKAGE_DIR)
	$(HELM) package $(HELM_CHART) --destination $(HELM_PACKAGE_DIR)

.PHONY: helm-install
helm-install:
	$(HELM) upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE) \
		--create-namespace \
		--set-string image.repository=$(IMAGE_REPOSITORY) \
		--set-string image.tag=$(IMAGE_TAG)

.PHONY: helm-uninstall
helm-uninstall:
	$(HELM) uninstall $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)
