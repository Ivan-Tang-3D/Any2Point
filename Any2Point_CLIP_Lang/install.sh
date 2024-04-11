pip install -r requirements.txt

cd cpp/pointnet2_batch
python setup.py install
cd ../

cd pointops/
python setup.py install
cd ..

cd chamfer_dist
python setup.py install --user

pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
