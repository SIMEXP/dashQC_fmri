FROM debian:buster-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    ca-certificates \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installing and setting up miniconda
RUN curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-py37_4.8.2-Linux-x86_64.sh

ENV PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include/:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

RUN conda config --add channels conda-forge
RUN conda install -y python=3.8.2 \
                     scikit-learn=0.22.1 \
                     pillow=7.0.0 \
                     pandas=1.0.3 \
                     numpy=1.18.1 \
                     nibabel=3.1.0 \
                     nilearn=0.6.2 \
                     matplotlib=3.1.3 \
                     joblib=0.15.1 \
                     tqdm=4.46.1; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda build purge-all; sync && \
    conda clean -tipsy && sync

WORKDIR /dashqc
COPY . /dashqc

ENTRYPOINT ["python", "/dashqc/dashQC_fmri/fmriprep_report2.py"] 