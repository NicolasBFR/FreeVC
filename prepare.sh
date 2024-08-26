wget -O dataset.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU" -O wavlm/WavLM-Large.pt
rm -rf /tmp/cookies.txt
python3 downsample.py --in_zip dataset.zip
rm dataset.zip
python3 preprocess_flist.py
python3 preprocess_spk.py
python3 preprocess_ssl.py
python3 preprocess_ssl.py
python3 train.py -c configs/freevc-nosr.json -m freevc
python3 convert.py --hpfile logs/freevc/config.json --ptfile logs/freevc/G_0.pth --txtpath convert.txt --outdir outputs/freevc
