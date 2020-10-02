python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/wserver/D0/Dataset/Kitti/SceneFlow/data_scene_flow/testing' \
                  --test_list='./lists/kitti2015_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/kitti2015_final.pth'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/wserver/D0/Dataset/Kitti/Stereo/2012/data_stereo_flow/testing' \
                  --test_list='./list/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/kitti2012_final.pth'