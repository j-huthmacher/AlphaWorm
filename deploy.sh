#sudo apt-get install sshpass rsync -y

echo "--Please enter IFI-CIP username (without @ and suffix)"
read username
address="${username}@remote.cip.ifi.lmu.de"
echo "--Enter IFI-CIP Password"
read -s password
echo "--Please enter project target directory name"
read dir
full_dir="/work/${dir}/"

rsync -av --progress --update ./scripts/start.sh ${address}:~${full_dir}
rsync -av --progress --update ./dev/* ${address}:~${full_dir}/dev/

ssh ${address} "sed -i 's|REPLACE|.$full_dir|g' .${full_dir}/start.sh"

ssh ${address} sbatch --partition=All --cpus-per-task=4 --job-name=${dir} ./work/${dir}/start.sh

"""

sudo apt-get install sshpass rsync -y

echo "--Please enter IFI-CIP username (without @ and suffix)"
read username
address="${username}@remote.cip.ifi.lmu.de"
echo "--Enter IFI-CIP Password"
read -s password
echo "--Please enter project target directory name"
read dir
full_dir="/work/${dir}/"

sshpass -p ${password} rsync -av --progress --update ./scripts/start.sh ${address}:~${full_dir}
sshpass -p ${password} rsync -av --progress --update ./dev/* ${address}:~${full_dir}/dev/

sshpass -p ${password} ssh ${address} "sed -i 's|REPLACE|.$full_dir|g' .${full_dir}/start.sh"

sshpass -p ${password} ssh ${address} sbatch --partition=All --cpus-per-task=4 --job-name=${dir} ./work/${dir}/start.sh
"""