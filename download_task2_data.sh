%%bash
cat << 'EOS' > download_task2_data.sh
#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/dcase2025t2/dev_data/raw
pushd data/dcase2025t2/dev_data/raw > /dev/null
for m in ToyCar ToyTrain bearing fan gearbox slider valve; do
  [ -d "$m" ] || { echo "dev_$m"; wget -q https://zenodo.org/records/15097779/files/dev_${m}.zip; unzip -q dev_${m}.zip; rm dev_${m}.zip; }
done
popd > /dev/null

mkdir -p data/dcase2025t2/eval_data/raw
pushd data/dcase2025t2/eval_data/raw > /dev/null
for m in AutoTrash HomeCamera ToyPet ToyRCCar BandSealer Polisher ScrewFeeder CoffeeGrinder; do
  [ -d "$m/train" ] || { echo "train_$m"; wget -q https://zenodo.org/records/15392814/files/eval_data_${m}_train.zip; unzip -q eval_data_${m}_train.zip; rm eval_data_${m}_train.zip; }
  [ -d "$m/test" ]  || { echo "test_$m";  wget -q https://zenodo.org/records/15519362/files/eval_data_${m}_test.zip;  unzip -q eval_data_${m}_test.zip;  rm eval_data_${m}_test.zip; }
done
popd > /dev/null
EOS