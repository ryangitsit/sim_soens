NUMBERS=$(seq 1 4)

for NUM in ${NUMBERS}
do
	sbatch -J JOB_NAME_${NUM} --export=VARIABLE=${NUM} enki_job.slurm
done