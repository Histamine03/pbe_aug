## Paint-by-Example Augmentation

### Data preparing
- https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view 에서 pretrained pbe model 다운로드
- ./pretrained_models 로 다운로드 받은 모델 이동

```
pbe_aug
├── gookbang
│  ├── pbe_aug_res
│  ├── video
│  │  ├── A-15
│  │  │  ├── xxx.mp4
│  │  │  ├── xxx.txt
│  │  │  ├── . . .
│  │  ├── A-sunrise
│  │  ├── ...
│  │  ├── B-sunrise
│  ├── video2
│  │  ├── Aregion-09
│  │  │  ├── xxx.mp4
│  │  │  ├── xxx.txt
│  │  │  ├── . . .
│  │  ├── Aregion-12-1
│  │  ├── ...
│  │  ├── Aregion 15-2
```

### 실행
```
git clone https://github.com/cnaigithub/pbe_aug.git
cd pbe_aug
conda env create -f environment.yaml
conda activate pbe_aug
python pbe_aug.py 
```