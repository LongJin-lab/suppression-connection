# Exploring Basic Connections: Robust Alternative to Residual Connection
Code for the paper "Exploring Basic Connections: Robust Alternative to Residual Connection."

## Requirements

- comet_ml

## Training Commands
### ImageNet
To train the ConvNeXt-S with the residual connection (baseline):
```python
./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 128 --model convnext_conver_small --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Res --sched cosine --epochs 300 --lr 0.001 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings Default --IniDecay 0
```
To train the ConvNeXt-S with the suppression connection (ours):
```python
./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 128 --model convnext_conver_small --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ResShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --sched cosine --epochs 300 --lr 0.001 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0
```
To train the ConvNeXt-S-N with the residual connection (baseline):
```python
./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 256 --model convnext_conver_small_narrow --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Res --sched cosine --epochs 90 --lr 0.002 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings Default
```
To train the ConvNeXt-S-N with the suppression connection (ours):
```python
./distributed_train1.sh 8 /tmp/_datasets/imagenet/ -b 256 --model convnext_conver_small_narrow --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ResShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --sched cosine --epochs 90 --lr 0.002 --opt AdamW --model-ema-decay 0.9999 --opt-eps 5e-9 --weight-decay 0.05 --train-interpolation bicubic --warmup-epochs 20 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --cooldown-epochs 0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --drop-path 0.1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0
```

To train the LeViT-256-D with the residual connection (baseline):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 768 --model levit_256d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Default --sched cosine --epochs 90 --lr 1.5e-3 --opt AdamW --model-ema-decay 0.99996 --opt-eps 1e-8 --weight-decay 0.025 --clip-grad 0.01 --clip-mode agc --momentum 0.9 --lr-noise-pct 0.67 --lr-noise-std 1.0 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --cooldown-epochs 0 --patience-epochs 10 --decay-rate 0.1 --train-interpolation bicubic --warmup-epochs 10 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --settings Default --IniDecay 0.7
```
To train the LeViT-256-D with the suppression connection (ours):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 768 --model levit_256d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ShareExpDecayLearnDecay --sched cosine --epochs 90 --lr 1.5e-3 --opt AdamW --model-ema-decay 0.99996 --opt-eps 1e-8 --weight-decay 0.025 --clip-grad 0.01 --clip-mode agc --momentum 0.9 --lr-noise-pct 0.67 --lr-noise-std 1.0 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --cooldown-epochs 0 --patience-epochs 10 --decay-rate 0.1 --train-interpolation bicubic --warmup-epochs 10 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0.7
```


To train the LeViT-128 with the residual connection (baseline):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 1536 --model levit_128  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Default --sched cosine --epochs 90 --lr 3e-3 --opt AdamW  --model-ema-decay 0.99996 --opt-eps 1e-8 --weight-decay 0.025 --clip-grad 0.01 --clip-mode agc --momentum 0.9 --lr-noise-pct 0.67 --lr-noise-std 1.0 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --cooldown-epochs 0 --patience-epochs 10 --decay-rate 0.1 --train-interpolation bicubic --warmup-epochs 10 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --settings Default --IniDecay 0.7
```
To train the LeViT-128 with the suppression connection (ours):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 1536 --model levit_128  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ShareExpDecayLearnDecay --sched cosine --epochs 90 --lr 3e-3 --opt AdamW --model-ema-decay 0.99996 --opt-eps 1e-8 --weight-decay 0.025 --clip-grad 0.01 --clip-mode agc --momentum 0.9 --lr-noise-pct 0.67 --lr-noise-std 1.0 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --cooldown-epochs 0 --patience-epochs 10 --decay-rate 0.1 --train-interpolation bicubic --warmup-epochs 10 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --amp --native-amp --dist-bn reduce --pin-mem -j 4 --checkpoint-hist 1 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0.7
```

To train the PreResNet50 with the residual connection (baseline):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 256 --model convernetv2_50d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Default --sched cosine --epochs 90 --lr 0.8 --amp --dist-bn reduce --warmup-epochs 10 --cooldown-epochs 0 --pin-mem -j 4 --settings Default --IniDecay 0.7
```
To train the PreResNet50 with the suppression connection (ours):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 256 --model convernetv2_50d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ShareExpDecayLearnDecay --sched cosine --epochs 90 --lr 0.8 --amp --dist-bn reduce --warmup-epochs 10 --cooldown-epochs 0 --pin-mem -j 4 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0.7
```

To train the PreResNet38 with the residual connection (baseline):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 256 --model convernetv2_38d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Default --sched cosine --epochs 90 --lr 0.8 --amp --dist-bn reduce --warmup-epochs 10 --cooldown-epochs 0 --pin-mem -j 4 --settings Default --IniDecay 0.7
```
To train the PreResNet38 with the suppression connection (ours):
```python
./distributed_train1.sh 8  /tmp/_datasets/imagenet/ -b 256 --model convernetv2_38d  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1ShareExpDecayLearnDecay --sched cosine --epochs 90 --lr 0.8 --amp --dist-bn reduce --warmup-epochs 10 --cooldown-epochs 0 --pin-mem -j 4 --settings ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --IniDecay 0.7
```
### Adversarial Training
To train the WRN-58-1 with the residual connection (baseline):
```python
python3 trainTuning.py --data-dir  ./datasets/  --log-dir ./log/Default --attack-step  0.00784313725490196  --attack-eps  0.03137254901960784  --settings Default --unsup-fraction  0.7  --LSE --ls  0
--adv-eval-freq  20  --coesB  -1 0  --learn
 False  --IniRes  False  --Mask  False  --IniCh  8  --data cifar10 --batch-size  512  --model  nrn-58-1-swish-learn  --num-adv-epochs  110  --lr  0.2  --scheduler  step  --beta  6.0  --attack  linf-pgd  --IniDecay  0.2
 --CoesLR  0.001  --desc Default
```
To train the WRN-58-1 with the suppression connection (ours):
```python
python3 trainTuning.py --data-dir  ./datasets/  --log-dir ./log/ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3 --attack-step  0.00784313725490196  --attack-eps  0.03137254901960784  --settings ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3 --unsup-fraction  0.7  --LSE --ls  0
--adv-eval-freq  20  --coesB  -1 0  --learn
 False  --IniRes  False  --Mask  False  --IniCh  8  --data cifar10 --batch-size  512  --model  nrn-58-1-swish-learn  --num-adv-epochs  110  --lr  0.2  --scheduler  step  --beta  6.0  --attack  linf-pgd  --IniDecay  0.2
 --CoesLR  0.001  --desc ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3
```
### CIFAR
To train the PreResNet110 (4x) with the residual connection (baseline) on CIFAR100:
```python
python3 Train.py --arch  ZeroSAny110settings   --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --eps_iter 0.01 --nb_iter 7  --notes  SumBase_ab1  --givenA   1 0   --givenB  -1 0  --lr 0.1 --bs 128 --opt SGD --dataset cifar100 --sche cos --steps  2  --eps  0.031  --IniDecay  0.07  --settings BnReluConv_ConvStride2ResLike --CoesLR  0.04  --save_path ./runs/cifar100/ --eps  0.031  --ConverOrd  1  --epoch 200 --warm 0 
```
To train the PreResNet110 (4x) with the suppression connection (ours) on CIFAR100:
```python
python3 Train.py --arch  ZeroSAny110settings   --minimizer None --rho 0.5 --eta 0.01 --ini_stepsize 1 --eps_iter 0.01 --nb_iter 7  --notes  SumBase_ab1  --givenA   1 0   --givenB  -1 0  --lr 0.1 --bs 128 --opt SGD --dataset cifar100 --sche cos --steps  2  --eps  0.031  --IniDecay  0.07  --settings BnReluConv_ConvStride2ResLike_ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3 --CoesLR  0.04  --save_path ./runs/cifar100/ --eps  0.031  --ConverOrd  1  --epoch 200 --warm 0 
```
