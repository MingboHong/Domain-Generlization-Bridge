import os
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import albumentations as A

@TRANSFORMS.register_module()
class AlbuDomainAdaption(BaseTransform):
    """Apply Albu domain adaption methods

    """

    def __init__(self,
                 domain_adaption_type: str = 'ALL',
                 target_dir: str = None,
                 p: float = 0.5,
                is_city: bool = False,
                is_voc:bool = False,
                is_weather:bool = False,) -> None:
        self.domain_adaption_type = domain_adaption_type
        self.target_dir = target_dir
        self.p = p
        self.is_city = is_city
        self.is_voc = is_voc
        self.is_weather = is_weather
        if self.is_city:
            self.target_list = [
                    os.path.join(root, file)
                    for root, dirs, files in os.walk(self.target_dir)
                    for file in files
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
            print(len(self.target_list))
        elif self.is_voc:
            self.target_list = []
            for voc_dir in self.target_dir:
           
                trainval_path = os.path.join(voc_dir, 'ImageSets', 'Main', 'trainval.txt')
                
                with open(trainval_path, 'r') as f:
                    image_ids = [line.strip() for line in f if line.strip()]
                
                for image_id in image_ids:
                    img_path = os.path.join(voc_dir, 'JPEGImages', image_id + '.jpg')
                    if os.path.isfile(img_path):
                        self.target_list.append(img_path)
        elif self.is_weather:
            self.target_list = []
            txt_path = self.target_dir  
            
            jpeg_dir = os.path.abspath(os.path.join(os.path.dirname(txt_path), '../../', 'JPEGImages'))
            #print(jpeg_dir)
            with open(txt_path, 'r') as f:
                image_ids = [line.strip() for line in f if line.strip()]

            for image_id in image_ids:
                img_path = os.path.join(jpeg_dir, image_id + '.jpg')
                if os.path.isfile(img_path):
                    self.target_list.append(img_path)
            #print(len(self.target_list))
        else:
            self.target_list = [os.path.join(self.target_dir, target_image) for target_image in os.listdir(self.target_dir)]
        assert self.domain_adaption_type in ["HistogramMatching", "FDA", "PixelDistributionAdaptation", 'ALL']
        assert len(self.target_list) > 0

    def transform(self, results: dict) -> dict:

        if self.domain_adaption_type == "HistogramMatching":
            aug = A.Compose([A.HistogramMatching(self.target_list, p=self.p)])
        elif self.domain_adaption_type == "FDA":
            aug = A.Compose([A.FDA(self.target_list, p=self.p)])
        elif self.domain_adaption_type == "PixelDistributionAdaptation":
            aug = A.Compose([A.PixelDistributionAdaptation(self.target_list, p=self.p)])
        else:
            aug = A.Compose(
                [A.OneOf(
                    [A.HistogramMatching(self.target_list, p=1.0),
                     A.FDA(self.target_list, p=1.0),
                     A.PixelDistributionAdaptation(self.target_list, p=1.0)])], p=self.p)

        adaption_result = aug(image=results['img'])['image']
        results['img'] = adaption_result

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'domain adaption type={self.domain_adaption_type}, '
        repr_str += f'target dir={self.target_dir}, '
        repr_str += f'p={self.p})'
        return repr_str
