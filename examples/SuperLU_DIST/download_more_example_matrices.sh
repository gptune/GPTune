#!/bin/bash

## This script is to get some more example matrix inputs from SuitSparse Matrix Collection* (https://sparse.tamu.edu/).
## * Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663

cd superlu_dist/EXAMPLE/

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si2.tar.gz
tar xvf Si2.tar.gz
mv Si2/Si2.mtx .
rmdir Si2

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/SiH4.tar.gz
tar xvf SiH4.tar.gz
mv SiH4/SiH4.mtx .
rmdir SiH4

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/SiNa.tar.gz
tar xvf SiNa.tar.gz
mv SiNa/SiNa.mtx .
rmdir SiNa

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Na5.tar.gz
tar xvf Na5.tar.gz
mv Na5/Na5.mtx .
rmdir Na5

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/benzene.tar.gz
tar xvf benzene.tar.gz
mv benzene/benzene.mtx .
rmdir benzene

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si10H16.tar.gz
tar xvf Si10H16.tar.gz
mv Si10H16/Si10H16.mtx .
rmdir Si10H16

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si5H12.tar.gz
tar xvf Si5H12.tar.gz
mv Si5H12/Si5H12.mtx .
rmdir Si5H12

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/SiO.tar.gz
tar xvf SiO.tar.gz
mv SiO/SiO.mtx .
rmdir SiO

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga3As3H12.tar.gz
tar xvf Ga3As3H12.tar.gz
mv Ga3As3H12/Ga3As3H12.mtx .
rmdir Ga3As3H12

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/GaAsH6.tar.gz
tar xvf GaAsH6.tar.gz
mv GaAsH6/GaAsH6.mtx .
rmdir GaAsH6

wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/H2O.tar.gz
tar xvf H2O.tar.gz
mv H2O/H2O.mtx .
rmdir H2O

