#!/bin/bash
# 完整用户画像生成脚本示例

# 设置基本参数
INPUT_FILE="/home/wangkuang/workshop/git/User_Simulator/dataset/data/test_data/test_data.jsonl"
OUTPUT_FILE="./output/complete_profiles.jsonl"
MAX_COUNT=100  # 测试时只处理100条数据
NUM_WORKERS=32

# 创建输出目录
mkdir -p ./output

echo "开始生成完整用户画像..."
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "最大处理数量: $MAX_COUNT"
echo "并行线程数: $NUM_WORKERS"

# 运行主程序
python generate_complete_profile.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_count $MAX_COUNT \
    --num_workers $NUM_WORKERS \
    --show_examples 5

echo "处理完成！"
echo "输出文件已保存到: $OUTPUT_FILE"

# 显示输出文件的前几行
echo "输出文件预览:"
head -n 3 "$OUTPUT_FILE" 