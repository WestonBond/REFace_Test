
##### EXPERIMENTAL #####

# Set variables
name="003_000"
Results_dir="results_video/${name}"
Base_dir="results_video"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=4

CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/last.ckpt"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"


CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "${Results_dir}" \
    --target_video "examples/FaceSwap/002.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "examples/FaceSwap/006_frames/frame_0000.png" \
    --Base_dir "${Base_dir}" \
    --scale 3 \
    --ddim_steps 50 

    

