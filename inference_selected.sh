name="Swap_outs"
Results_dir="examples/FaceSwap/${name}/results"
Base_dir="examples/FaceSwap/${name}/Outs"
Results_out="examples/FaceSwap/${name}/results/results" 
device=5

CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/last.ckpt"

declare -a swaps=(
    #"003_000"
    #"006_002"
    #"982_004"
    #"002_006"
    #"990_008"
    "005_010"
    "026_012"
    "790_014"
    "344_020"
    "489_022"
)

for swap in "${swaps[@]}"; do
    target_path="examples/FaceSwap/${swap%_*}_frames"
    source_id=${swap#*_}
    temp_source_dir="examples/FaceSwap/${name}/temp_source"
    mkdir -p "${temp_source_dir}"
    cp "examples/FaceSwap/${source_id}_frames/frame_0000.png" "${temp_source_dir}/"
    
    echo "Processing: Target=${target_path}, Source=${temp_source_dir}"

    CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_selected.py \
        --outdir "${Results_dir}/${swap}" \
        --target_folder "${target_path}" \
        --config "${CONFIG}" \
        --ckpt "${CKPT}" \
        --src_folder "${temp_source_dir}" \
        --Base_dir "${Base_dir}/${swap}" \
        --n_samples 1 \
        --scale 3 \
        --ddim_steps 50

    rm -rf "${temp_source_dir}"
done