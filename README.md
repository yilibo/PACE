# PACE: Process-Aligned Compositional Encoder for Video ActionRecognition

PACE is a video action recognition framework built around the **Process-Aligned Compositional Encoder for Video ActionRecognition** method:
- frozen CLIP visual backbone
- trajectory-guided token construction
- masked trajectory relation encoding
- sparse trajectory-to-visual write-back
- frame-text bidirectional alignment for classification

## Environment

```bash
conda create -n pace python=3.10
conda activate pace
pip install -r requirements.txt
```

## Data

Set your video dataset root and trajectory metadata root:

```bash
export DATA_DIR=<PATH_TO_VIDEO_DATASET>
export PACE_PT_DATA=<PATH_TO_TRAJECTORY_METADATA>
```

Trajectory metadata is read from:
- `DATA.PATH_TO_PACE_PT_DATA`

Trajectory preprocessing under `point_tracking/` uses CoTracker query initialization with:
- regular grid queries
- motion-guided random queries
- custom query JSON files

## Training

Use the sample launcher:

```bash
bash scripts/pace.sh pace_ssv2_train
```

Or run manually:

```bash
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    tools/run_net.py --init_method env:// --new_dist_init \
    --cfg configs/pace/$CONFIG_TO_USE.yaml \
    DATA.PATH_TO_DATA_DIR $DATA_DIR \
    DATA.PATH_TO_PACE_PT_DATA $PACE_PT_DATA \
    MODEL.MODEL_NAME PACE \
    MODEL.METHOD.FREEZE_CLIP True \
    POINT_INFO.SAMPLING_TYPE hybrid_motion \
    POINT_INFO.NUM_POINTS_TO_SAMPLE 256
```

## Paper-Aligned Defaults

Current defaults are aligned to the PACE experimental setting:
- input size `224x224`
- trajectory points `256`
- trajectory module injected into last `6` CLIP blocks
- frozen CLIP backbone
- cross-entropy classification objective over bidirectional frame-text alignment scores

## Configs

Main configs are under `configs/pace/`:
- `pace_k400_train.yaml`
- `pace_ssv2_train.yaml`
- `pace_ucf101_zeroshot_eval.yaml`
- `pace_hmdb51_zeroshot_eval.yaml`
