#!/usr/bin/env bash
# =============================================================================
# 一键训练：训练源模型 -> 源无关域适配（Office-Home）
#
# 运行前请先执行一次 bash scripts/link_data.sh 建立数据软链接。
#
# 用法（在项目根目录下执行）：
#   bash scripts/run.sh
#
# 后台运行（断开 SSH 也不会中断）：
#   nohup bash scripts/run.sh > run_all.log 2>&1 &
#
# 只想跑某一部分时，修改下方「阶段开关」或「任务列表」即可。
# =============================================================================

# 任一命令出错就立即退出；使用未定义变量也会报错；管道中任一环节失败视为失败
set -euo pipefail

# -----------------------------------------------------------------------------
# 一、基本配置
# -----------------------------------------------------------------------------

# 数据集名，对应项目里的 dataset/<DATASET>/ 目录
DATASET="OfficeHomeDataset"

# Python 解释器（如果用 conda 虚拟环境，可改成该环境下 python 的完整路径）
PYTHON="python"

# 日志输出目录（每个任务一个日志文件）
LOG_DIR="runs/logs"

# -----------------------------------------------------------------------------
# 二、训练参数
# -----------------------------------------------------------------------------

# 使用的 GPU 编号（只对 main.py 生效；trainSource.py 内部固定使用 0 号卡）
GPU=0

# 源模型训练轮数（trainSource.py 默认 20）
SOURCE_EPOCHS=20

# 域适配训练轮数（README 中 Office-Home 用的是 40）
ADAPT_EPOCHS=40

# -----------------------------------------------------------------------------
# 三、阶段开关：1 = 执行，0 = 跳过
# -----------------------------------------------------------------------------
RUN_SOURCE=1   # 训练源模型
RUN_ADAPT=1    # 域适配

# 源模型已存在时是否跳过训练（1 = 跳过，0 = 重新训练并覆盖）
SKIP_EXISTING_SOURCE=1

# -----------------------------------------------------------------------------
# 四、任务列表
# -----------------------------------------------------------------------------

# 要训练的源域（命令行用的小写名字，代码内部会转成 Art / Clipart / Product / Real World）
SOURCE_DOMAINS=(art clipart product realworld)

# 要跑的域适配任务，格式为 "源域:目标域"；不想跑的行可以在前面加 # 注释掉
ADAPT_PAIRS=(
    "art:clipart"
    "art:product"
    "art:realworld"
    "clipart:art"
    "clipart:product"
    "clipart:realworld"
    "product:art"
    "product:clipart"
    "product:realworld"
    "realworld:art"
    "realworld:clipart"
    "realworld:product"
)

# =============================================================================
# 以下一般不需要修改
# =============================================================================

# 切换到脚本所在目录（即项目根目录），因为代码用相对路径读 dataset/ 和写 runs/
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
mkdir -p "$LOG_DIR"

# 检查软链接是否已建立
for d in "Art" "Clipart" "Product" "Real World"; do
    if [ ! -d "dataset/$DATASET/$d" ]; then
        echo "[错误] 找不到 dataset/$DATASET/${d}，请先执行 bash scripts/link_data.sh"; exit 1
    fi
done

# 域名缩写，用于实验名，例如 art + clipart -> Ar2Cl
abbr() {
    case "$1" in
        art)       echo "Ar" ;;
        clipart)   echo "Cl" ;;
        product)   echo "Pr" ;;
        realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

# 源模型保存目录名
# main.py 读取的是 runs/Source_home/Source_<域名小写>/Source.pt，
# 而 realworld 在代码中会变成 "Real World"，小写后是 "real world"（带空格），
# 所以 realworld 的目录名必须是 "Source_real world"
source_dir_name() {
    if [ "$1" = "realworld" ]; then
        echo "Source_real world"
    else
        echo "Source_$1"
    fi
}

# -----------------------------------------------------------------------------
# 第 1 步：训练源模型
#   输出：runs/Source_home/Source_<域>/Source.pt
# -----------------------------------------------------------------------------
if [ "$RUN_SOURCE" = "1" ]; then
    echo "==> [1/2] 训练源模型"
    for s in "${SOURCE_DOMAINS[@]}"; do
        name="Source_home/$(source_dir_name "$s")"

        if [ "$SKIP_EXISTING_SOURCE" = "1" ] && [ -f "runs/$name/Source.pt" ]; then
            echo "  跳过 ${s}（已存在 runs/$name/Source.pt）"
            continue
        fi

        log="$LOG_DIR/source_$s.log"
        echo "  训练源域 ${s}，日志: $log"
        # tee：同时输出到屏幕和日志文件
        "$PYTHON" trainSource.py \
            --name "$name" \
            --dataset "$DATASET" \
            --source "$s" \
            --epochs "$SOURCE_EPOCHS" \
            2>&1 | tee "$log"
    done
fi

# -----------------------------------------------------------------------------
# 第 2 步：源无关域适配
#   输出：runs/OfficeHome/<Ar2Cl 等>/
#   注意：第一次运行会联网下载 CLIP 权重
# -----------------------------------------------------------------------------
if [ "$RUN_ADAPT" = "1" ]; then
    echo "==> [2/2] 域适配"
    for pair in "${ADAPT_PAIRS[@]}"; do
        s="${pair%%:*}"
        t="${pair##*:}"
        task="$(abbr "$s")2$(abbr "$t")"

        # 适配前确认对应的源模型已经训练好
        ckpt="runs/Source_home/$(source_dir_name "$s")/Source.pt"
        if [ ! -f "$ckpt" ]; then
            echo "  [错误] 缺少源模型 ${ckpt}，请先训练源域 $s"; exit 1
        fi

        log="$LOG_DIR/adapt_$task.log"
        echo "  ${task}（$s -> ${t}），日志: $log"
        "$PYTHON" main.py \
            --device "$GPU" \
            --name "OfficeHome/$task" \
            --epochs "$ADAPT_EPOCHS" \
            --dataset "$DATASET" \
            --source "$s" \
            --target "$t" \
            2>&1 | tee "$log"
    done
fi

echo "==> 全部完成"
