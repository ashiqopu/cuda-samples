
# CUDA Samples Docker Environment for Benchmarking
# Based on NVIDIA CUDA runtime with development tools
# Using CUDA 13.0 as required by cuda-samples

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
# If your build env has restrictive /tmp, redirect temp usage
ENV TMPDIR=/var/tmp

# Ensure /tmp exists and is correctly writable
RUN mkdir -p /tmp && chmod 1777 /tmp

# Base prerequisites for apt, TLS, and gnupg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg dirmngr apt-transport-https curl \
 && rm -rf /var/lib/apt/lists/*

# (Optional) Re-declare NVIDIA CUDA repo with keyring if `/tmp` errors persist
# RUN set -eux; \
#     mkdir -p /etc/apt/keyrings; \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
#       | gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg; \
#     echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg arch=amd64] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
#       > /etc/apt/sources.list.d/cuda.list; \
#     apt-get update

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    python3 \
    python3-pip \
    pkg-config \
    libfreeimage-dev \
    libglfw3-dev \
    libglew-dev \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libopenmpi-dev \
    openmpi-bin \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgles-dev \
    mesa-common-dev \
    wget \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Create glut.pc for pkg-config (freeglut3-dev doesn't provide it)
RUN mkdir -p /usr/lib/pkgconfig && \
    echo "prefix=/usr" > /usr/lib/pkgconfig/glut.pc && \
    echo "exec_prefix=\${prefix}" >> /usr/lib/pkgconfig/glut.pc && \
    echo "libdir=\${exec_prefix}/lib/x86_64-linux-gnu" >> /usr/lib/pkgconfig/glut.pc && \
    echo "includedir=\${prefix}/include" >> /usr/lib/pkgconfig/glut.pc && \
    echo "" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Name: glut" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Description: OpenGL Utility Toolkit (freeglut)" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Version: 3.0.0" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Requires: gl glu" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Libs: -L\${libdir} -lglut -lGL -lGLU" >> /usr/lib/pkgconfig/glut.pc && \
    echo "Cflags: -I\${includedir}" >> /usr/lib/pkgconfig/glut.pc

# Create a user with matching UID/GID from host (passed as build args)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=cudauser

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /cuda-samples

# Change ownership of the working directory
RUN chown -R ${USERNAME}:${USERNAME} /cuda-samples

# Switch to non-root user
USER ${USERNAME}

# Default command
CMD ["/bin/bash"]
