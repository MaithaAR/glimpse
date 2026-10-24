# GLIMPSE in a Linux container: Python 3.14, the exact locked libraries (CPU-only PyTorch), and the pinned photo
# weights downloaded and SHA-256-checked at build time, so the container runs with the network switched off.
# The CGMacros data is NOT inside the image (CC BY-NC-SA licence): mount your own copy to run the demo (see README).
FROM python:3.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# 1. libraries only (cached layer): exactly the versions in uv.lock, no development tools
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# 2. the project: code, settings, splits and the trained checkpoints (all small, all committed)
COPY src ./src
COPY configs ./configs
COPY splits ./splits
COPY checkpoints ./checkpoints
RUN uv sync --locked --no-dev

# 3. pinned ConvNeXt-Tiny weights (exact Hugging Face commit, SHA-256 verified); offline from here on
RUN /app/.venv/bin/glimpse download-weights
ENV HF_HUB_OFFLINE=1

# 4. run as an ordinary user; only logs/ and outputs/ are writable
RUN useradd --create-home glimpse && mkdir -p logs outputs && chown -R glimpse logs outputs
USER glimpse
ENTRYPOINT ["/app/.venv/bin/glimpse"]
CMD ["selftest"]
