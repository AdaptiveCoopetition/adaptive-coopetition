#!/bin/bash

#deepmath-2k-sample gsm8k-symbolic-2k-sample
#touch log/${model}-32b_baseline_gsm8k_symbolic-${dataset}.jsonl;
#
# thudm/glm-z1-32b-0414 (done)
# google/gemma-3-27b-it gpt-4o deepseek/deepseek-v3-0324  deepseek/deepseek-r1-distill-qwen-14b qwen/qwq-32b
# gsm8k-symbolic-200-sample deepmath-200-sample
# --continue_from_checkpoint CONTINUE_FROM_CHECKPOINT
# --continue_from_checkpoint=True
#
for dataset in deepmath-200-sample ; do
#		echo dataset=${dataset};
		echo "python3 demo.py --input=../data/${dataset}.jsonl"
		echo "                --output=../log/mad_${dataset}.jsonl"
                echo "                --parallelism=1 "
		echo "                --max_num_questions=1"
		echo "                --filter_output=''"	
		python3 mad_demo.py --input=../data/${dataset}.jsonl \
				--output=../log/mad_${dataset}_comp.jsonl \
				--parallelism=1 \
				--filter_output='';
	done;
