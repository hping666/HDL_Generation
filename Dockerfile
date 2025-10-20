FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 
 # Do not need use cuda, so just runtime image
 # if need compile CUDA extensions, use -devel- tag

# install micromamba (lightweight conda)
ARG MAMBA_VERSION=1.5.8
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates tini && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} \
    | tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1 && \
    rm -rf /var/lib/apt/lists/*

ARG ENV_NAME=hypergraph
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN mkdir -p $MAMBA_ROOT_PREFIX

WORKDIR /opt/app
COPY environment.yml /opt/app/environment.yml


RUN micromamba create -y -n $ENV_NAME -f environment.yml && micromamba clean -a -y

# 1) 安装 PyTorch cu117 官方 wheel（必须带 +cu117）
RUN micromamba run -n $ENV_NAME pip install \
  torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

# 2) 安装 PyG
RUN micromamba run -n $ENV_NAME pip install --no-cache-dir --prefer-binary \
  --only-binary=torch-scatter,torch-sparse,torch-cluster,torch-spline-conv \
  https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.2%2Bpt20cu117-cp310-cp310-linux_x86_64.whl \
  https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.18%2Bpt20cu117-cp310-cp310-linux_x86_64.whl \
  https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.3%2Bpt20cu117-cp310-cp310-linux_x86_64.whl \
  https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_spline_conv-1.2.2%2Bpt20cu117-cp310-cp310-linux_x86_64.whl

# 3) 最后装纯 Python 的 torch-geometric
RUN micromamba run -n $ENV_NAME pip install torch-geometric==2.6.0
RUN micromamba run -n $ENV_NAME pip install OhMyRunPod

ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-V"]
