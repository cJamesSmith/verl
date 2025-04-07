# Audio-Text Question Answer

## Dataset

1. Download [VGG-Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset from [hugginface](https://huggingface.co/datasets/Loie/VGGSound) (Because the download links for the VGG-Sound dataset is no longer available from its official website).

2. Unzip all the zipped files.

3. Use `ffmpeg` to extract `.wav` from `.mp4` files.

4. Download [avqa](https://mn.cs.tsinghua.edu.cn/avqa/) dataset (Since it is a very large file and most of it is unused in this project, you can download the useful files from my [onedrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/xwc_zju_edu_cn/EjQuS3LVq6NArUVewnD7WjcBUlutMdrKvI0yqZSO_UjhKA?e=cipHLa)).

5. Preprocess `avqa` dataset