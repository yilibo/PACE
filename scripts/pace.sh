export CONFIG_TO_USE=$1
export NUM_GPUS=1
export NUM_WORKERS=32
export MASTER_PORT=$(cat /dev/urandom | tr -dc '0-9' | fold -w 4 | head -n 1) 
export NUM_POINTS_TO_SAMPLE=256
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

# Infer dataset from config name unless user already provided DATASET.
if [[ -z "$DATASET" ]]; then
	if [[ $CONFIG_TO_USE == *"ssv2"* ]]; then
		export DATASET="ssv2"
	elif [[ $CONFIG_TO_USE == *"k400"* ]]; then
		export DATASET="k400"
	elif [[ $CONFIG_TO_USE == *"ucf"* ]]; then
		export DATASET="ucf101"
	elif [[ $CONFIG_TO_USE == *"hmdb"* ]]; then
		export DATASET="hmdb51"
	else
		export DATASET=$CONFIG_TO_USE
	fi
fi
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$CONFIG_TO_USE/$EXP_NAME/$SECONDAY_EXP_NAME
export DATA_DIR=/scratch1/pulkit/$DATASET

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
	tools/run_net.py --init_method env:// --new_dist_init \
	--cfg configs/pace/$CONFIG_TO_USE.yaml \
	WANDB.ID $WANDB_ID \
	WANDB.EXP_NAME $EXP_NAME \
	MASTER_PORT $MASTER_PORT \
	OUTPUT_DIR $OUTPUT_DIR \
	NUM_GPUS $NUM_GPUS \
	DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
	DATA.PATH_TO_DATA_DIR $DATA_DIR \
	DATA.PATH_TO_PACE_PT_DATA $PACE_PT_DATA \
	POINT_INFO.NAME $POINT_INFO_NAME \
	POINT_INFO.SAMPLING_TYPE hybrid_motion \
	POINT_INFO.NUM_POINTS_TO_SAMPLE $NUM_POINTS_TO_SAMPLE \
	MODEL.FEAT_EXTRACTOR clip \
	MODEL.MODEL_NAME PACE \
	MODEL.METHOD.CLIP_MODEL_NAME ViT-B/16 \
	MODEL.METHOD.FREEZE_CLIP True
	
