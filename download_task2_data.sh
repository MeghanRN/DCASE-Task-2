%%bash
cat << 'EOS' > download_task2_data.sh
#!/usr/bin/env bash
set -euo pipefail

# === Development data (7 machines) ===
mkdir -p data/dcase2025t2/dev_data/raw
pushd data/dcase2025t2/dev_data/raw > /dev/null
for m in ToyCar ToyTrain bearing fan gearbox slider valve; do
  if [ ! -d "$m" ]; then
    echo "Downloading dev_${m}.zip"
    wget -q "https://zenodo.org/records/15097779/files/dev_${m}.zip"
    unzip -q "dev_${m}.zip" && rm "dev_${m}.zip"
  fi
done
popd > /dev/null

# === Additional training + evaluation test (8 machines) ===
mkdir -p data/dcase2025t2/eval_data/raw
pushd data/dcase2025t2/eval_data/raw > /dev/null
for m in AutoTrash HomeCamera ToyPet ToyRCCar BandSealer Polisher ScrewFeeder CoffeeGrinder; do
  # train
  if [ ! -d "${m}/train" ]; then
    echo "Downloading eval_data_${m}_train.zip"
    wget -q "https://zenodo.org/records/15392814/files/eval_data_${m}_train.zip"
    unzip -q "eval_data_${m}_train.zip" && rm "eval_data_${m}_train.zip"
  fi
  # test
  if [ ! -d "${m}/test" ]; then
    echo "Downloading eval_data_${m}_test.zip"
    wget -q "https://zenodo.org/records/15519362/files/eval_data_${m}_test.zip"
    unzip -q "eval_data_${m}_test.zip" && rm "eval_data_${m}_test.zip"
  fi
done
popd > /dev/null

echo "Datasets downloaded."
EOS