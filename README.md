# suppression-connection
Code for the paper "Robustness Requires Different Structures Than Residual Connection"

## Requirements

- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```
pip install git+https://github.com/fra31/auto-attack
```
## Training Commands
To train the ConvNeXt-S with the residual connection (baseline):
```python
nohup ./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 128 --model convnext_conver_small --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Res --sched cosine --epochs 300 --lr 0.001 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings Default --IniDecay 0
```
To train the ConvNeXt-S with the suppression connection (ours):
```python
nohup ./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 128 --model convnext_conver_small --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ResShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --sched cosine --epochs 300 --lr 0.001 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0
```
To train the ConvNeXt-S-N with the residual connection (baseline):
```python
nohup ./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 256 --model convnext_conver_small_narrow --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Res --sched cosine --epochs 90 --lr 0.002 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings Default
```
To train the ConvNeXt-S-N with the suppression connection (ours):
```python
nohup ./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 256 --model convnext_conver_small_narrow --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ResShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --sched cosine --epochs 90 --lr 0.002 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0
```
