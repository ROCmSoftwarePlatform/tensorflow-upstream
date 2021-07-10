# Tensorflow ROCm port: Basic installation on CentOS

## Intro

These instructions provide a starting point for using the TensorFlow ROCm port on CentOS.

*Note*: it is recommended to start with a clean CentOS 7.8 system

## Install ROCm

Add the ROCm repository:  
```
export RPM_ROCM_REPO=http://repo.radeon.com/rocm/yum/3.7
```

Install misc pkgs:
```
# Enable extra repositories
yum --enablerepo=extras install -y epel-release

# Install required base build and packaging commands for ROCm
yum -y install \
    bc \
    cmake \
    cmake3 \
    dkms \
    dpkg \
    elfutils-libelf-devel \
    expect \
    file \
    gettext \
    gcc-c++ \
    git \
    libgcc \
    ncurses \
    ncurses-base \
    ncurses-libs \
    numactl-devel \
    numactl-libs \
    libssh \
    libunwind-devel \
    libunwind \
    llvm \
    llvm-libs \
    make \
    openssl \
    openssl-libs \
    openssh \
    openssh-clients \
    pciutils \
    pciutils-devel \
    pciutils-libs \
    python36 \
    python36-devel \
    pkgconfig \
    qemu-kvm \
    rpm \
    rpm-build \
    subversion \
    wget
```

Install ROCm pkgs:
```
# Add the ROCm package repo location
echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo

# Install the ROCm rpms
sudo yum clean all
sudo yum install -y rocm-dev
sudo yum install -y hipblas hipcub hipsparse miopen-hip miopengemm rccl rocblas rocfft rocprim rocrand
```

Ensure the ROCm target list is set up
```
bash -c 'echo -e "gfx803\ngfx900\ngfx906\ngfx908" >> $ROCM_PATH/bin/target.lst'
```

## Install required python packages


```
pip3.6 install --user \
    cget \
    pyyaml \
    pip \
    setuptools==39.1.0 \
    virtualenv \
    absl-py \
    six==1.10.0 \
    protobuf==3.6.1 \
    numpy==1.18.2 \
    scipy==1.4.1 \
    scikit-learn==0.19.1 \
    pandas==0.19.2 \
    gnureadline \
    bz2file \
    wheel==0.29.0 \
    portpicker \
    werkzeug \
    grpcio \
    astor \
    gast \
    termcolor \
    h5py==2.8.0 \
    keras_preprocessing==1.0.5
```

## Install TensorFlow

```
# Install ROCm manylinux WHL 
wget <location of WHL file>
pip3.6 install --user ./tensorflow*linux_x86_64.whl
```

## Quick sanity test

```
cd ~ && git clone -b cnn_tf_v1.15_compatible https://github.com/tensorflow/benchmarks.git
python3.6 ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50
```
