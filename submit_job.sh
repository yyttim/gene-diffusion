#!/bin/bash
set -e

# Job configuration
JOB_NAME="train_600m_efficient"
NODE_NAME=$1
if [ -z "$NODE_NAME" ]; then
    echo "Usage: $0 <node_name>"
    exit 1
fi

# Check execution environment
if [[ $(hostname) =~ cyclone ]]; then
    EXECUTE_DIRECTLY=true
else
    EXECUTE_DIRECTLY=false
fi

# Setup paths
PROJECT_ROOT="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/${JOB_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${JOB_NAME}.${DSUB_JOB_ID:-$$}.out"
CONFIG_FILE="${PROJECT_ROOT}/configs/train_4gpu_600m.yaml"

# Create execution script
EXEC_SCRIPT="/tmp/${JOB_NAME}_exec_$$.sh"
cat > ${EXEC_SCRIPT} << 'EOF'
#!/bin/bash
set -e

# Load modules
module purge
module load compilers/gcc/9.3.0
module load cuda/11.8
module load compilers/cuda/11.8
module load cudnn/8.6.0_cuda11.x

# CUDA environment
export CUDA_HOME=/soft/cuda/11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0

# PyTorch settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Disable warnings
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=disabled
export WANDB_SILENT=true

# GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Python path
export PYTHONPATH="${PYTHONPATH}:PROJECT_ROOT"

# Activate conda environment
source /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate gene-diffusion

cd PROJECT_ROOT

# Distributed training
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=4

# Launch training
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py \
    --config CONFIG_FILE \
    --output_dir OUTPUT_DIR 2>&1 | tee $LOG_FILE
EOF

# Replace placeholders
sed -i "s|PROJECT_ROOT|${PROJECT_ROOT}|g" ${EXEC_SCRIPT}
sed -i "s|CONFIG_FILE|${CONFIG_FILE}|g" ${EXEC_SCRIPT}
sed -i "s|OUTPUT_DIR|${OUTPUT_DIR}|g" ${EXEC_SCRIPT}
sed -i "s|LOG_FILE|${LOG_FILE}|g" ${EXEC_SCRIPT}

chmod +x ${EXEC_SCRIPT}

# Execute or submit
if [ "$EXECUTE_DIRECTLY" = true ]; then
    ${EXEC_SCRIPT}
else
    # Submit to HPC
    dsub \
        -s ${EXEC_SCRIPT} \
        -N ${JOB_NAME}_${TIMESTAMP} \
        -A chengxuz_g1 \
        -R "cpu=32;gpu=4;mem=120000" \
        -eo ${LOG_DIR}/%J.%I.err.log \
        -oo ${LOG_FILE} \
        --constraint "$NODE_NAME"
    
    echo "Job submitted. Log: ${LOG_FILE}"
fi

# Cleanup
rm -f ${EXEC_SCRIPT} 