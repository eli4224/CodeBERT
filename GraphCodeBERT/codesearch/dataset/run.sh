if [ ! -d dataset ]; then
    mkdir dataset
fi
cd dataset

# Download and extract .zip files for each language
for lang in python java ruby javascript go php; do
    if [ ! -f $lang.zip ]; then
        wget https://zenodo.org/record/7857872/files/$lang.zip
        unzip $lang.zip
    fi
done

rm -f *.pkl

# Run preprocess.py if train.jsonl is not present
if [ ! -f train.jsonl ]; then
    cd ..
    python3 preprocess.py
    cd dataset
fi

# Clean up any unnecessary directories
rm -r */final
cd ..
