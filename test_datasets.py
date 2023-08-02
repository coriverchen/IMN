import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from options import arguments
opt=arguments()
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class testdataset(Dataset):
    def __init__(self, transforms_=None):

        self.transform = transforms_


        self.files = sorted(glob.glob(opt.testpath + "/*.png"))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):

        return len(self.files)

transform_val = T.Compose([
    T.CenterCrop(opt.cropsize_test),
    T.ToTensor(),
])



# Test data loader
testloader = DataLoader(
    testdataset(transforms_=transform_val),
    batch_size=2,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    drop_last=True
)