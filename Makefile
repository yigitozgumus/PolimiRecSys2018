
test:
	python main.py --j configs/user_knn.json -e -l

cython:
	python base/Cython/compileCython.py base/Cython/Similarity.pyx build_ext --inplace

create_c:
	cd base/Cython;python compileCython.py Similarity.pyx build_ext --inplace;cd ../..