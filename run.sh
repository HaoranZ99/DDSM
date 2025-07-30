python src/main.py --model DDSM --dataset Cora --lr 1e-2 --weight_decay 1e-3 --dropout 0.7 --num_layers 7 --alpha 0.1612340973926313 --beta 0.00012969664091482674 --eta 0.10065809037686718 --type pagerank
python src/main.py --model DDSM --dataset CiteSeer --lr 2e-4 --weight_decay 1e-2 --dropout 0.5 --num_layers 9 --alpha 0.247010247174746 --beta 0.01948065841777242 --eta 0.4068777717305878 --k 32 --type pagerank
python src/main.py --model DDSM --dataset PubMed --lr 5e-2 --weight_decay 5e-5 --dropout 0.4 --num_layers 9 --alpha 0.29492017490067485 --beta 0.017824882072039836 --eta 0.7403550209402463 --type pagerank
python src/main.py --model DDSM --dataset CoraFull --lr 5e-4 --weight_decay 1e-4 --dropout 0.3 --num_layers 4 --alpha 0.1986826436011211 --beta 0.003152803921720163 --eta 0.15975696942306628 --type pagerank
python src/main.py --model DDSM --dataset CS --lr 2e-4 --weight_decay 1e-4 --dropout 0.7 --num_layers 10 --alpha 0.9952896196967153 --beta 0.4719268475360486 --eta 0.4555863975992882 --type pagerank
python src/main.py --model DDSM --dataset Physics --lr 5e-3 --weight_decay 1e-5 --dropout 0.2 --num_layers 8 --alpha 0.9625381932508495 --beta 0.7869003037397945 --eta 0.09208177002791566 --k 32 --type pagerank
python src/main.py --model DDSM --dataset Cornell --lr 2e-2 --weight_decay 1e-4 --dropout 0.6 --num_layers 8 --alpha 0.955980482516899 --beta 0.000321769312722665 --eta 0.772363104338448 --type pagerank
python src/main.py --model DDSM --dataset Texas --lr 1e-2 --weight_decay 5e-2 --dropout 0.5 --num_layers 2 --alpha 0.6272854649750712 --beta 0.12082671927745636 --eta 0.017230896942647258 --type pagerank
python src/main.py --model DDSM --dataset Wisconsin --lr 1e-2 --weight_decay 1e-2 --dropout 0.7 --num_layers 4 --alpha 0.9839341973830087 --beta 0.0004528353757804679 --eta 0.20120789863727626 --type pagerank
python src/main.py --model DDSM --dataset Chameleon --lr 2e-3 --weight_decay 1e-4 --dropout 0.2 --num_layers 1 --alpha 0.0003440617421545583 --beta 0.0382820368437065 --eta 0.23935233124957578 --type pagerank
python src/main.py --model DDSM --dataset WikiCS --lr 1e-2 --weight_decay 1e-4 --dropout 0.3 --num_layers 2 --alpha 0.1497812185268177 --beta 0.4313244957838067 --eta 0.23859501747224765 --type pagerank

python src/main.py --model DDSM --dataset Cora --lr 2e-3 --weight_decay 5e-3 --dropout 0.6 --num_layers 7 --alpha 0.2893459896828756 --beta 0.03601843089617686 --eta 0.6566379658944893 --type vanilla
python src/main.py --model DDSM --dataset CiteSeer --lr 1e-4 --weight_decay 1e-2 --dropout 0.6 --num_layers 9 --alpha 0.3084165771496061 --beta 0.04773437894928838 --eta 0.8 --type vanilla
python src/main.py --model DDSM --dataset PubMed --lr 5e-2 --weight_decay 5e-5 --dropout 0.2 --num_layers 7 --alpha 0.3553310216542821 --beta 0.0009949819894185148 --eta 0.0012270068401629651 --k 128 --type vanilla
python src/main.py --model DDSM --dataset CoraFull --lr 5e-4 --weight_decay 1e-4 --dropout 0.3 --num_layers 4 --alpha 0.18935867007314153 --beta 0.0017255197846292036 --eta 0.9984122359916181 --type vanilla
python src/main.py --model DDSM --dataset CS --lr 2e-4 --weight_decay 1e-4 --dropout 0.6 --num_layers 10 --alpha 0.7413818916519425 --beta 0.19306253641385607 --eta 0.9609242505128572 --type vanilla
python src/main.py --model DDSM --dataset Physics --lr 2e-2 --weight_decay 5e-5 --dropout 0.2 --num_layers 8 --alpha 0.6936319039917529 --beta 0.6417626731490246 --eta 0.3900103547479792 --k 32 --type vanilla
python src/main.py --model DDSM --dataset Cornell --lr 5e-3 --weight_decay 1e-3 --dropout 0.7 --num_layers 8 --alpha 0.9990606087917011 --beta 0.0011961059180565525 --eta 0.8509809348891433 --type vanilla
python src/main.py --model DDSM --dataset Texas --lr 2e-2 --weight_decay 5e-2 --dropout 0.2 --num_layers 2 --alpha 0.9995401273140909 --beta 0.15469593509098964 --eta 0.6036013590190957 --type vanilla
python src/main.py --model DDSM --dataset Wisconsin --lr 1e-2 --weight_decay 1e-2 --dropout 0.6 --num_layers 8 --alpha 0.9998564226158576 --beta 0.0367522389164296 --eta 0.8599333715819009 --type vanilla
python src/main.py --model DDSM --dataset Chameleon --lr 2e-3 --weight_decay 1e-4 --dropout 0.5 --num_layers 1 --alpha 0.012873238132032654 --beta 0.001256879275779138 --eta 0.9965321993191051 --type vanilla
python src/main.py --model DDSM --dataset WikiCS --lr 2e-3 --weight_decay 5e-5 --dropout 0.2 --num_layers 2 --alpha 0.21857480812310676 --beta 0.17898437795452088 --eta 0.6827488881947905 --k 32 --type vanilla

python src/main.py --model DDSM --dataset Cora --lr 5e-3 --weight_decay 1e-3 --dropout 0.7 --num_layers 7 --alpha 0.14575391475451902 --beta 0.03429686202302662 --eta 0.4330313562393987 --type heat
python src/main.py --model DDSM --dataset CiteSeer --lr 2e-4 --weight_decay 1e-2 --dropout 0.5 --num_layers 9 --alpha 0.2459666388187181 --beta 0.041678955758225694 --eta 0.9449054803939642 --k -1 --type heat
python src/main.py --model DDSM --dataset PubMed --lr 5e-2 --weight_decay 1e-4 --dropout 0.3 --num_layers 7 --alpha 0.3808238853565508 --beta 0.000361044651926279 --eta 0.999901959487431 --type heat
python src/main.py --model DDSM --dataset CoraFull --lr 1e-3 --weight_decay 1e-3 --dropout 0.1 --num_layers 9 --alpha 0.44927552400277837 --beta 0.19346008492054667 --eta 0.962254325088725 --type heat
python src/main.py --model DDSM --dataset CS --lr 1e-4 --weight_decay 5e-4 --dropout 0.1 --num_layers 1 --alpha 0.748613721527777 --beta 0.086507979852483 --eta 0.86498196335835 --type heat
python src/main.py --model DDSM --dataset Physics --lr 1e-2 --weight_decay 5e-4 --dropout 0.5 --num_layers 8 --alpha 0.7151524928947757 --beta 0.5192657951269122 --eta 0.41475102519433865 --type heat
python src/main.py --model DDSM --dataset Cornell --lr 5e-3 --weight_decay 5e-4 --dropout 0.6 --num_layers 8 --alpha 0.999193421533292 --beta 0.000102477978449895 --eta 0.125559455794954 --type heat
python src/main.py --model DDSM --dataset Texas --lr 1e-2 --weight_decay 5e-2 --dropout 0.2 --num_layers 2 --alpha 0.5913646970329148 --beta 0.04496350227464454 --eta 0.304957042017305 --k -1 --type heat
python src/main.py --model DDSM --dataset Wisconsin --lr 2e-2 --weight_decay 1e-2 --dropout 0.7 --num_layers 8 --alpha 0.9680228624613482 --beta 0.037411996394730154 --eta 0.33534596406701606 --type heat
python src/main.py --model DDSM --dataset Chameleon --lr 1e-2 --weight_decay 5e-5 --dropout 0.6 --num_layers 1 --alpha 0.019827330239729227 --beta 0.019425359280147287 --eta 0.9129374936477161 --type heat
python src/main.py --model DDSM --dataset WikiCS --lr 2e-3 --weight_decay 1e-4 --dropout 0.1 --num_layers 2 --alpha 0.28242648322078157 --beta 0.9177593380308569 --eta 0.5 --type heat --k 16
