from dataloader.data import get_training_set, get_test_set
from torch.utils.data import DataLoader


train_set = get_training_set("/media/wserver/D0/Dataset/Kitti/SceneFlow/data_scene_flow/training/",
                             "./lists/kitti2015_train.list", [240, 528], False, False, True, False)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True, drop_last=True)

for iter, batch in enumerate(training_data_loader):
    print(batch.shape)
