wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
mkdir emojis
unzip -q openmoji-72x72-color.zip -d ./emojis
pip install -q tensorflow==2.4