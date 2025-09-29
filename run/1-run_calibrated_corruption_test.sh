#!/bin/bash

set -e
cd "$(dirname "$0")/.."

MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

TASKS=(
    "operator_induction_interleaved_text"
    "operator_induction_text"
)

SHOTS=(4 6 8)
ROLLOUTS=3
SAMPLES=200

echo "Starting comprehensive evaluation batch..."
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Shots: ${SHOTS[*]}"
echo "Working directory: $(pwd)"
echo ""

TOTAL=$((${#MODELS[@]} * ${#TASKS[@]} * ${#SHOTS[@]}))
COUNT=0

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for shots in "${SHOTS[@]}"; do
            COUNT=$((COUNT + 1))
            
            # Get model type
            if [[ "$model" == *"Qwen3"* ]]; then
                model_type="qwen3"
            elif [[ "$model" == *"Qwen2.5-VL"* ]]; then
                model_type="qwen25"
            elif [[ "$model" == *"InternVL"* ]]; then
                model_type="internvl"
            else
                model_type="llama3"
            fi
            
            # Get data directory
            if [[ "$task" == "google_analogy" ]]; then
                data_dir="./data"
            else
                data_dir="./VL-ICL"
            fi
            
            model_short=$(basename "$model")
            
            echo "[$COUNT/$TOTAL] $model_short | $task | ${shots}-shot"
            log_file="results/run_${model_short}_${task}_${shots}shot.log"
            
            cmd="python scripts/comprehensive_evaluation.py \
                --model_name \"$model\" \
                --model_type \"$model_type\" \
                --dataset \"$task\" \
                --data_dir \"$data_dir\" \
                --n_shot $shots \
                --num_samples $SAMPLES \
                --num_rollouts $ROLLOUTS"
            
            echo "Command: $cmd"
            echo "Log: $log_file"
            
            if eval "$cmd" 2>&1 | tee "$log_file"; then
                echo "SUCCESS"
            else
                echo "FAILED"
            fi
            echo ""
        done
    done
done

echo "All runs completed!"
echo "Results saved in: results/"
find results -name "comprehensive_*.json" -newer . 2>/dev/null | sort