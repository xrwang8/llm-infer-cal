# Deployment

This project ships a single production container image:

- Rust `llm-infer-cal-web` serves the API on port `8080`.
- The React frontend is built into static files and served by the same process.
- The frontend calls `/api/*` on the same origin by default.

## Build Locally

```sh
make build
```

## Build The Container Image

```sh
make docker-build IMAGE_REPOSITORY=llm-infer-cal IMAGE_TAG=0.1.0
```

Run it locally:

```sh
make docker-run IMAGE_REPOSITORY=llm-infer-cal IMAGE_TAG=0.1.0
```

Then open `http://127.0.0.1:8080`.

## Helm

Lint and render the chart:

```sh
make helm-lint
make helm-template IMAGE_REPOSITORY=llm-infer-cal IMAGE_TAG=0.1.0
```

Package the chart:

```sh
make helm-package
```

Install or upgrade:

```sh
make helm-install \
  HELM_RELEASE=llm-infer-cal \
  HELM_NAMESPACE=llm-infer-cal \
  IMAGE_REPOSITORY=llm-infer-cal \
  IMAGE_TAG=0.1.0
```

Port-forward when Ingress is disabled:

```sh
kubectl -n llm-infer-cal port-forward svc/llm-infer-cal 8080:80
```

Then open `http://127.0.0.1:8080`.

## Common Overrides

```sh
helm upgrade --install llm-infer-cal charts/llm-infer-cal \
  --namespace llm-infer-cal \
  --create-namespace \
  --set image.repository=registry.example.com/llm-infer-cal \
  --set image.tag=0.1.0 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=llm-infer-cal.example.com
```
