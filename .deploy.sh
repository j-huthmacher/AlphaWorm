
ssh lmu_cip "mkdir -p ~/ddpg_adapated_td3"
ssh lmu_cip "ls ~/ddpg_adapated_td3/"
echo "--Please enter project target directory name"
read dir
full_dir="~/ddpg_adapated_td3/${dir}"

ssh lmu_cip "mkdir -p ${full_dir}"

rsync -av -e ssh ./scripts/start.sh lmu_cip:${full_dir}
rsync -av -e ssh --exclude='*.pyc' --exclude='__pycache__' --exclude='dev/models/*' --exclude='dev/logs/*' --exclude='dev/notebooks/*' ./dev lmu_cip:${full_dir}

ssh lmu_cip "sed -i 's|REPLACE|$full_dir|g' ${full_dir}/start.sh"
echo "Start from ${full_dir}"
ssh lmu_cip sbatch --partition=All --cpus-per-task=4 ${full_dir}/start.sh
