#!/usr/bin/env bash
# =============================================================================
# 建立 Office-Home 数据软链接（只需执行一次，重复执行也没问题）
#
# 用法（在项目根目录下执行）：
#   bash scripts/link_data.sh
#
# 执行后项目内的目录结构：
#   dataset/OfficeHomeDataset/
#   ├── Art             -> /hyperai/input/input0/Art                 <- 图片文件夹
#   ├── Art.txt         -> /hyperai/input/input0/Art_list.txt        <- 图片列表
#   ├── Art.txt.bak                                                  <- 项目原来的列表（备份）
#   ├── ...
#   ├── Real World      -> /hyperai/input/input0/RealWorld           <- 注意名字带空格
#   └── Real World.txt  -> /hyperai/input/input0/RealWorld_list.txt
#
# 代码中 IMAGEFILE = "dataset"，所以：
#   列表路径：dataset/OfficeHomeDataset/<域>.txt
#   图片路径：dataset/OfficeHomeDataset/<域>/<列表中每行的第一列>
#
# 恢复项目原来的列表：删掉 <域>.txt 软链接，把 <域>.txt.bak 改回 <域>.txt
# =============================================================================

# 任一命令出错就立即退出；使用未定义变量也会报错
set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------

# 云端原始数据所在目录
DATA_SRC="/hyperai/input/input0"

# 数据集名，对应项目里的 dataset/<DATASET>/ 目录
DATASET="OfficeHomeDataset"

# 每个域的映射，格式为 "云端图片文件夹|云端列表文件|项目中使用的域名"
# 项目中的域名必须和代码里 args.domains 一致（Art / Clipart / Product / Real World）
DOMAINS=(
    "Art|Art_list.txt|Art"
    "Clipart|Clipart_list.txt|Clipart"
    "Product|Product_list.txt|Product"
    "RealWorld|RealWorld_list.txt|Real World"
)

# 是否使用云端的列表文件：1 = 用云端的 *_list.txt，0 = 用项目自带的 txt
USE_CLOUD_LIST=1

# =============================================================================
# 以下一般不需要修改
# =============================================================================

# 切换到脚本所在目录（即项目根目录），保证下面的相对路径正确
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

LINK_ROOT="dataset/$DATASET"
if [ ! -d "$LINK_ROOT" ]; then
    echo "[错误] 找不到 ${LINK_ROOT}，请确认在项目根目录下执行"; exit 1
fi

# 建立一个软链接；如果目标位置已有真实文件/目录则报错，不去覆盖
make_link() {
    local src="$1" dst="$2"
    if [ ! -e "$src" ]; then
        echo "[错误] 云端文件不存在: ${src}"; exit 1
    fi
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "[错误] ${dst} 已存在且不是软链接，请手动检查"; exit 1
    fi
    # -s 软链接，-f 已存在则替换，-n 不跟随已有的链接（防止把链接建到目录内部）
    # 链接目标用绝对路径，这样无论从哪里访问链接都能找到数据
    ln -sfn "$src" "$dst"
    echo "  $dst -> $src"
}

for item in "${DOMAINS[@]}"; do
    # 按 | 拆成三段
    IFS="|" read -r img_dir list_file name <<< "$item"

    # ---- 1. 图片文件夹 ----
    make_link "$DATA_SRC/$img_dir" "$LINK_ROOT/$name"

    # ---- 2. 图片列表 ----
    if [ "$USE_CLOUD_LIST" = "1" ]; then
        txt="$LINK_ROOT/$name.txt"
        # 项目原来的 txt 是真实文件时，先改名备份（只备份一次）
        if [ -f "$txt" ] && [ ! -L "$txt" ]; then
            if [ -e "$txt.bak" ]; then
                echo "[错误] ${txt} 和 ${txt}.bak 同时存在，请手动处理"; exit 1
            fi
            mv "$txt" "$txt.bak"
            echo "  备份 $txt -> $txt.bak"
        fi
        make_link "$DATA_SRC/$list_file" "$txt"
    fi
done

# 类名文件：main.py 从 dataset/<DATASET>/classname.txt 读取类名，行号即标签编号
make_link "$DATA_SRC/classname.txt" "$LINK_ROOT/classname.txt"

# -----------------------------------------------------------------------------
# 检查列表内容是否符合代码要求
#   代码按空格拆分每一行：第一列 = 图片相对路径（相对于 <域> 文件夹），第二列 = 标签
# -----------------------------------------------------------------------------
echo "检查列表内容..."
for item in "${DOMAINS[@]}"; do
    IFS="|" read -r img_dir list_file name <<< "$item"
    txt="$LINK_ROOT/$name.txt"

    # tr -d '\r'：去掉 Windows 换行的 \r
    first_line=$(head -n 1 "$txt" | tr -d '\r')
    first_img=$(echo "$first_line" | awk '{print $1}')

    # 检查 1：每行至少有两列（路径 + 标签），且第二列是整数
    bad=$(tr -d '\r' < "$txt" | awk 'NF>0 && (NF<2 || $2 !~ /^[0-9]+$/)' | head -n 1)
    if [ -n "$bad" ]; then
        echo "[错误] ${txt} 格式不对，需要「路径 标签」，问题行: $bad"; exit 1
    fi

    # 检查 2：第一张图能否按代码的拼接方式找到
    if [ ! -f "$LINK_ROOT/$name/$first_img" ]; then
        echo "[错误] 按列表第一行拼出的路径不存在: $LINK_ROOT/$name/$first_img"
        echo "       列表第一行: $first_line"
        echo "       第一列需要是相对于 $img_dir 文件夹的路径，如 Alarm_Clock/00001.jpg"
        exit 1
    fi

    # 检查 3：标签编号是否和项目原来的一致（用 .bak 备份对照）
    #   取路径中图片所在的文件夹名作为类别名，比较「类别名 -> 标签」的对应关系
    #   顺序不一致时训练不会报错，但结果会完全错误，所以这里必须拦下
    if [ -f "$txt.bak" ]; then
        mismatch=$(
            { tr -d '\r' < "$txt.bak" | sed 's/^/REF /'; tr -d '\r' < "$txt" | sed 's/^/NEW /'; } |
            awk '
                NF < 3 { next }
                { n = split($2, p, "/"); cls = p[n-1] }
                $1 == "REF" { ref[cls] = $3; next }
                (cls in ref) && ref[cls] != $3 { print cls ": 项目为 " ref[cls] "，云端为 " $3; exit }
            '
        )
        if [ -n "$mismatch" ]; then
            echo "[错误] ${txt} 的标签顺序和项目原来的不一致：$mismatch"
            echo "       如需恢复项目原来的列表，把 USE_CLOUD_LIST 改成 0 并按文件开头的说明恢复"
            exit 1
        fi
    fi

    echo "  $txt 检查通过（$(grep -c . "$txt") 行）"
done

echo "软链接建立完成"
