# Docker Directory

This directory stores Docker-specific infrastructure files only.

Full usage instructions are documented in the root README:

- `/Users/terrylimax/medical-rag-reranker/README.md`
  - `Docker Compose (Nginx + App + Static + Broker)`
  - `Docker Image (Single-Container CLI)`

File map:

- `/Users/terrylimax/medical-rag-reranker/docker/nginx/`
  - external nginx gateway image and reverse-proxy config
- `/Users/terrylimax/medical-rag-reranker/docker/static/`
  - dedicated static-site image and frontend assets

Key files:

- `/Users/terrylimax/medical-rag-reranker/docker/nginx/nginx.conf`
- `/Users/terrylimax/medical-rag-reranker/docker/static/default.conf`
- `/Users/terrylimax/medical-rag-reranker/docker/static/site/index.html`
