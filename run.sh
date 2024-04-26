file_in="BDM_870.json"
dir_img_gen="output"

split=1
for((i=0;i<${split};i++));do
    CUDA_VISIBLE_DEVICES=${i} python inference.py ${file_in} ${dir_img_gen} ${i} ${split}
done