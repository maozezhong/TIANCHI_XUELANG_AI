## 比赛信息
- 本次大赛要求选手开发算法模型，通过布样影像，基于对布样中疵点形态、长度、面积以及所处位置等的分析，判断瑕疵的种类 。通过探索布样疵点精确智能诊断的优秀算法，提升布样疵点检验的准确度，降低对大量人工的依赖，提升布样疵点质检的效果和效率。
- [比赛链接](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.70731756uJzvoZ&raceId=231666)

## 文件说明
- code : 存放所有相关代码的文件夹
    - main.py :  主函数， 运行该函数进行模型的训练及预测，可以得到最终结果
    - split_good_bad.py : 将原始文件按照有无瑕疵分成good和bad两个文件夹
    - extract_xml.py : 将所有xml文件提取出来放在xml文件夹下面
    - DataAugmentForTrain.py : 对训练数据进行线下增强
    - DataAugmentForValid.py ： 将增强后的数据作为validation
    - del_copy_for_train.py ： 讲增强后的数据及原始数据全都copy到data_for_train文件夹中用于训练
    - merge.py : 融合最终的多个结果
- data : 存放原始数据文件，[官方数据下载地址](https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165261.5678.2.164f419dba7Pjd&raceId=231666)
- submit : 存放提交文件

## 操作说明
- step1 : 手动解压原始数据压缩文件，得保证解压后的文件名没有乱码！！！！！！！
- step2 : **手动在keras源码中修改插值方式,keras默认插值resize会出现波纹**：
    - 在~/anaconda2/lib/python2.7/site-packages/keras_preprocessing/imge.py的第33行后面加入 'antialias': pil_image.ANTIALIAS
    - 变成：
        '''
	_PIL_INTERPOLATION_METHODS = {
		'nearest': pil_image.NEAREST,
		'bilinear': pil_image.BILINEAR,
		'bicubic': pil_image.BICUBIC,
		'antialias': pil_image.ANTIALIAS,   #added by mao
	    }
        '''
- step3 : 运行main.py

## 思路说明
- 本次比赛我们团队分两个支路进行，我负责的这块使用的是keras，队友负责的那块使用的是pytorch, 所以会有两个模型训练预测part1和part2，这个在main.py里面有注释
- part1部分：
    - 数据增强线下：每张图片扩充到两张，加入了裁剪，改变亮度，加噪声，cutout等方式，当然增强的时候利用了xml文件的信息，保证了框也随之变化. 
    - 数据增强线上：使用keras内置的增强方法，开启了旋转，镜像，shear等
    - 模型：densenet
    - 修改loss， [参考](https://spaces.ac.cn/archives/4493)
- part2部分：
    - 数据增强：只有线上，使用pytorch内置的增强方式
    - 模型：resnet152
- 单模92左右，3个densenet模型融合线上能达到93.8%左右
- 比赛最终用的是3个densenet和1个resnet(队友pytorch训练)出来的结果进行融合，达到了**线上94.9%,9/2403**的成绩
- **注意，模型初始化用的是imagenet预训练权值，在开始模型训练前会自行下载**

## 随机性
- 线下线上增强是随机增强的，这会有一个随机性，结果可能会在线上最好成绩附近波动

## 另
- 针对目标检测的数据增强，数据预处理等相关脚本见：[Data_Preprocess_For_CV](https://github.com/maozezhong/Data_Preprocess_For_CV)
- part2部分为队友用pytorch，finetune resnet152, 在线镜像增强，输入resize到800，原始数据，训练得到。
- 上传代码未包括队友的部分
- 如果想达到949的精确度，可以在main.py代码里面将模型改为resnet152再训练一个模型，融合一下
- 感觉再多训练几个模型融合线上还能提高 = =
- [复赛代码（已更新）](https://github.com/maozezhong/TIANCHI_XUELANG_AI_2)
- 比赛数据
	- [初赛part1](https://pan.baidu.com/s/1KoZcXKCCaWLWfGc5Q4gCjg), 密码: 2qdn
	- [初赛part2](https://pan.baidu.com/s/1c0o7WKm-ETPcIyF6JPS3Wg),  密码: jq9a
	- [复赛](https://pan.baidu.com/s/1wuA0VT7E7SBtkrvarfPCcw), 密码: vyj9
