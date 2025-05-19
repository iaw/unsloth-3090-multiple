# ┌────────────────────────────────────────────────────────────────────────────┐
# 1) Base CUDA image + non‑interactive apt                                    │
# └────────────────────────────────────────────────────────────────────────────┘
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# ┌────────────────────────────────────────────────────────────────────────────┐
# 2) Install system deps + Python 3.12 + venv support                          │
# └────────────────────────────────────────────────────────────────────────────┘
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      git \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      libffi-dev \
      libssl-dev \
      libxml2-dev \
      libxslt1-dev \
      zlib1g-dev \
      libjpeg-dev \
      libpng-dev \
      libglib2.0-dev \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgtk-3-dev \
      libdbus-glib-1-dev \
      libsystemd-dev \
      libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ┌────────────────────────────────────────────────────────────────────────────┐
# 3) Create & activate venv                                                     │
# └────────────────────────────────────────────────────────────────────────────┘
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ┌────────────────────────────────────────────────────────────────────────────┐
# 4) Set workdir & copy wheelhouse + requirements                              │
# └────────────────────────────────────────────────────────────────────────────┘
WORKDIR /app
COPY wheelhouse        /wheelhouse
COPY requirements.txt  .

# ┌────────────────────────────────────────────────────────────────────────────┐
# 5) Upgrade pip & install all wheels (no deps, from /wheelhouse)             │
# └────────────────────────────────────────────────────────────────────────────┘
RUN pip install --upgrade pip wheel setuptools && \
    pip install \
      --prefer-binary \
      --no-index \
      --find-links=/wheelhouse \
      --no-deps \
      -r requirements.txt

# ┌────────────────────────────────────────────────────────────────────────────┐
# 5b) Install bitsandbytes and FA-2 from the same wheelhouse                            │
# └────────────────────────────────────────────────────────────────────────────┘
RUN pip install \
      --prefer-binary \
      --no-index \
      --find-links=/wheelhouse \
      bitsandbytes

# now install FlashAttention
RUN pip install \
      --prefer-binary \
      --no-index \
      --find-links=/wheelhouse \
      flash_attn==2.7.4.post1

# ┌────────────────────────────────────────────────────────────────────────────┐
# 6) Copy in your patched source checkouts                                      │
# └────────────────────────────────────────────────────────────────────────────┘
COPY unsloth/     /app/unsloth/
COPY unsloth_zoo/ /app/unsloth_zoo/

# ┌────────────────────────────────────────────────────────────────────────────┐
# 7) Install your patched packages in editable mode                            │
# └────────────────────────────────────────────────────────────────────────────┘
RUN pip install -e /app/unsloth \
                -e /app/unsloth_zoo

# ┌────────────────────────────────────────────────────────────────────────────┐
# 8) Default entrypoint                                                        │
# └────────────────────────────────────────────────────────────────────────────┘
CMD ["/bin/bash"]
