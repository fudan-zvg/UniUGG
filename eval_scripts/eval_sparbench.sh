export MODEL=internvl2
export TASK=sparbench
export SUFFIX=internvl2_full
export PRETRAIN=data/InternVL2-4B

accelerate launch --main_process_port 51123 --num_processes 8 -m lmms_eval \
  --model ${MODEL} --tasks ${TASK} --batch_size 1 --log_samples --log_samples_suffix ${SUFFIX} \
  --output_path ./logs/ --model_args pretrained=${PRETRAIN},dtype=bf16,attn_implementation=flash-attn