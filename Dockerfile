ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# LD_LIBRARY_PATH sauber initialisieren (keine Self-Refs in derselben Zeile)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# optional: später erweitern – hier ist es bereits definiert, daher unkritisch:
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/ros/${ROS_DISTRO}/lib

SHELL ["/bin/bash","-c"]

# System + ROS (OpenCV via APT für cv_bridge – kein pip-OpenCV!)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config git curl wget \
      python3 python3-pip python3-dev \
      python3-colcon-common-extensions python3-rosdep python3-vcstool \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      python3-opencv libopencv-dev \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-vision-msgs \
      ros-${ROS_DISTRO}-visualization-msgs \
      ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-ament-index-python \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-image-geometry \
      ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
      ros-${ROS_DISTRO}-tf-transformations \
      ros-${ROS_DISTRO}-interactive-markers \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

# (rosdep nur nötig, wenn du unten rosdep install nutzt)
RUN rosdep init || true && rosdep update

RUN python3 -m pip install -U pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade \
    --ignore-installed "transforms3d>=0.4.1"

############################################################################
# RUN python3 - <<'PY'
# import sys, tf_transformations, transforms3d
# print("tf_transformations:", tf_transformations.__file__)
# print("transforms3d:", transforms3d.__version__, transforms3d.__file__)
# print("sys.path[:3]:", sys.path[:3])
# PY
############################################################################

# PyTorch CUDA (cu121)
RUN pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

WORKDIR /ros2_ws

# PyPI-Requirements (nur PyPI, keine ROS-Pakete)
COPY requirements.txt constraints.txt /tmp/pip/
RUN if [ -s /tmp/pip/requirements.txt ]; then \
      pip install --no-cache-dir -r /tmp/pip/requirements.txt -c /tmp/pip/constraints.txt; \
    else \
      echo "No extra PyPI deps"; \
    fi

# Ultralytics ohne deps (cv2 kommt aus APT, torch/vision hast du separat)
RUN pip install --no-cache-dir --no-deps "ultralytics>=8.1.0,<9"

# Jetzt erst der Code (Caching)
COPY ros2_ws/src src

# Optional: System-Abhängigkeiten deiner ROS-Pakete via rosdep
# RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
#     rosdep install --from-paths src --ignore-src -r -y

# Build
RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install --symlink-install

# Komfort
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.sh" >> /root/.bashrc && \
#     echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=33



