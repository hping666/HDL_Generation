FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 
 # Do not need use cuda, so just runtime image
 # if need compile CUDA extensions, use -devel- tag

# install micromamba (lightweight conda)
ARG MAMBA_VERSION=1.5.8
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates tini && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} \
    | tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1 && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y iverilog

# install Ollama CLI
RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 11434

ARG ENV_NAME=HDL_Generation
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN mkdir -p $MAMBA_ROOT_PREFIX

WORKDIR /opt/app
COPY environment.yml /opt/app/environment.yml


RUN micromamba create -y -n $ENV_NAME -f environment.yml && micromamba clean -a -y

ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

RUN echo '#!/bin/bash\n\
set -e\n\
export OLLAMA_MODELS=/workspace/ollama_models\n\
if [ ! -d "$OLLAMA_MODELS" ]; then\n\
  echo "[INFO] Creating Ollama models directory at $OLLAMA_MODELS"\n\
  mkdir -p "$OLLAMA_MODELS"\n\
fi\n\
echo "[INFO] Starting Ollama service..."\n\
ollama serve &\n\
sleep 2\n\
echo "[INFO] Ollama started. Entering shell."\n\
exec "$@"' > /opt/app/start.sh && chmod +x /opt/app/start.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/app/start.sh", "bash"]
