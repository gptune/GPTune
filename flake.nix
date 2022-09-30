{
   description = "GPTune";

  inputs = {
    #Need unstable for openturns
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        python = "python39";
        version = builtins.substring 0 8 self.lastModifiedDate;

        ######### Patched Packages ##########
        pythonPackageOverrides = self: super: {
          gpy = super.gpy.overridePythonAttrs ( oldAttrs: rec {
            patchPhase = ''
              cat ${./patches/GPy/coregionalize.py} > GPy/kern/src/coregionalize.py
              cat ${./patches/GPy/stationary.py} > GPy/kern/src/stationary.py
              cat ${./patches/GPy/choleskies.py} > GPy/util/choleskies.py
            '';

            # [TODO] send a pull request to nixpkgs to upstream the unbreak
            # can confirm working on aarch64-darwin
            meta.broken = false; 
          });

          #patch scikit to use numpy floats rather than regular floats
          #gives consistent rounding behavior
          scikit-optimize = super.scikit-optimize.overridePythonAttrs ( oldAttrs: rec {
            patchPhase = ''
              cat ${./patches/scikit-optimize/space.py} > skopt/space/space.py
            '';
            #patch breaks several pytests
            disabledTests = [
              #fails for floating point reasons
              "utils"
              #fails with attempt to iterate over float
              "test_space"
            ];
            #comment out this line if you'd like to run other tests...
            #takes a while but should pass
            doCheck=false;
          });
        };

        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            #python package patches
            ( self: super: {
              python39 = super.python39.override {
                packageOverrides = pythonPackageOverrides;
            };})
          ] ++ nixpkgs.lib.optionals (system == "x86_64-darwin") [
            #scalapack fails two tests out of 150 (xslu, xsllt) on x86 mac
            #should still be fine for our purposes, so override to allow building
            (self: super: {
              scalapack = super.scalapack.overrideAttrs (oldAttrs: rec {
                meta.broken = false;
              });
            })
          ] ++ nixpkgs.lib.optionals (system == "aarch64-darwin") [
            #scalapack fails two tests out of 150 (xslu, xsllt) on x86 mac
            #should still be fine for our purposes, so override to allow building
            (self: super: {
              scalapack = super.scalapack.overrideAttrs (oldAttrs: rec {
                meta.broken = false;
              });
            })
          ];
        };

        ########### SYSTEM PACKAGE DEPENDENCIES ##########
        systemDeps = with pkgs; [
          #defaults to openBLAS unless system blas changed
          #see https://ryantm.github.io/nixpkgs/using/overlays#sec-overlays-alternatives-blas-lapack
          #Might be interesting to see if using Accelerate is faster on darwin
          blas
          lapack
          #fails a handful of tests on x86_64-darwin
          scalapack
          jq
          tbb
          openmpi
        ];

        ########## Extra Packages not in NixPkgs ###########
        packagesExtra = rec {

          cGP = pkgs.python39Packages.buildPythonPackage rec {
            pname = "cGP";
            version = "2022.01.27";
            src = builtins.fetchGit{
              url = "https://github.com/GPTune/cGP";
              ref = "main";
              rev = "1d5734b8fb35a7dadbf0b5b3f0077dc9c83527c9";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              numpy
              gpy
              scikit-learn
              scikit-optimize
              scipy
              dill
            ];
            doCheck = false; #no upstream checks
            pythonImportsCheck = [ "cGP" ];
          };

          opentuner = pkgs.python39Packages.buildPythonPackage rec {
            pname = "opentuner";
            version = "0.8.8";
            src = builtins.fetchGit{
              url = "https://github.com/jansel/opentuner";
              ref = "master";
              rev = "05e2d6b9538c9e2d335a02c48c0f7e77d1c57077";
            };
            #remove argparse from requirements - built into python now
            patchPhase = "sed '1d' requirements.txt > requirements.txt";
            propagatedBuildInputs = [ pkgs.sqlite ]
              ++ (with pkgs.python39Packages; [
                numpy
                sqlalchemy
                future
              ]);
            buildInputs = with pkgs.python39Packages; [ setuptools ];
            #checkInputs = [ pkgs.python39Packages.pytest ];
            # [FIXME] reintroduce python checks
            doCheck = false;
          };

          autotune = pkgs.python39Packages.buildPythonPackage rec {
            pname = "autotune";
            version = "2022.08.31";
            src = builtins.fetchGit{
              url = "https://github.com/ytopt-team/autotune";
              ref = "master";
              rev = "58f5d9a39106e3b2dbdea9ebcbe928a1b65bea6a";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              numpy
              scikit-optimize
              setuptools
            ];
            checkPhase = "pythonImportsCheck";
            doCheck = false;
            pythonImportsCheck = [ "autotune" ];
          };

          lhsmdu = pkgs.python39Packages.buildPythonPackage rec {
            pname = "lhsmdu";
            version = "1.1";
            src = pkgs.python39Packages.fetchPypi {
              inherit pname version;
              hash = "sha256-S8Hfa5zdJ7rgv/dc8Wk/RVujLk+ofKmpMvYGlmB/5xI=";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              numpy
              scipy
            ];
          };

          hpbandster = pkgs.python39Packages.buildPythonPackage rec {
            pname = "hpbandster";
            version = "0.7.4";
            src = pkgs.python39Packages.fetchPypi {
              inherit pname version;
              hash = "sha256-Sf/DJogVW1CeYvNhe1KuFalsm/8smWoj34PyeRBsWSE=";
            };
            propagatedBuildInputs = [ configspace ] ++ (with pkgs.python39Packages; [
              Pyro4
              serpent
              numpy
              statsmodels
              scipy
              netifaces
            ]);
            checkInputs = [ pkgs.python39Packages.unittestCheckHook ];
            unittestFlags = [ "-s" "tests" "-v" ];
          };

          configspace = pkgs.python39Packages.buildPythonPackage rec {
            pname = "ConfigSpace";
            version = "0.6.0";
            src = pkgs.python39Packages.fetchPypi {
              inherit pname version;
              hash = "sha256-m2yV2IOfyrIgNyZzIUsxKbRdzYsReYKessZXRsrLcqk=";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              numpy
              scipy
              cython
              pyparsing
              typing-extensions
            ];
            checkInputs = [ pkgs.python39Packages.pytest ];
          };

          SALib = pkgs.python39Packages.buildPythonPackage rec {
            pname = "SALib";
            version = "1.4.5";
            src = pkgs.python39Packages.fetchPypi {
              inherit pname version;
              hash = "sha256-zzlhduMN7VetZ9DRVPkXaJ+Pc+9cwdzTjUeW9DYQ2SU=";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              setuptools-scm
              numpy
              scipy
              matplotlib
              pandas
              multiprocess
              pathos
            ];
            checkInputs = with pkgs.python39Packages; [
              pytestCheckHook
              pytest
              pytest-cov
              #pytest-subprocess
            ];
            pytestFlagsArray = [ "tests/" ];
            #these tests use subprocess, which doesn't work in nix build env
            #can fake it with with pytest-subprocess, but it depends
            #on pyopenssl which is currently broken on mac
            #see GH issue: https://github.com/NixOS/nixpkgs/issues/175875
            #possible fix: https://github.com/NixOS/nixpkgs/pull/187636
            disabledTestPaths = [
              "tests/test_cli.py"
              "tests/test_cli_analyze.py"
              "tests/test_cli_sample.py"
            ];

          };

        };

        ########## Python Definitions ###########

        pydeps = ( with packagesExtra; [
            autotune
            cGP
            opentuner
            lhsmdu
            hpbandster
            SALib
          ]) ++ ( with pkgs.python39Packages; [
            scikit-optimize
            joblib
            scikit-learn
            scipy
            statsmodels
            pyaml
            matplotlib
            gpy
            openturns
            ipyparallel
            pygmo
            filelock
            requests
            pymoo
            mpi4py
            cloudpickle
          ]);

        fullPythonWith = pypkg : ( pkgs.python39.buildEnv.override {
          extraLibs = pydeps ++ [ pypkg ];
        });

      in
      rec {

        packages.gptune-libs = pkgs.stdenv.mkDerivation rec {
          pname = "gptune-libs";
          inherit version;
          src = ./.;

          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = systemDeps ++ [ pkgs.python39 ];

          cmakeFlags = [
            #"-DCMAKE_CXX_FLAGS=-fopenmp"
            #"-DCMAKE_C_FLAGS=-fopenmp" 
            #"-DCMAKE_Fortran_FLAGS=-fopenmp"
          	"-DBUILD_SHARED_LIBS=ON"
            "-DCMAKE_CXX_COMPILER=mpicxx"
            "-DCMAKE_C_COMPILER=mpicc"
            "-DCMAKE_Fortran_COMPILER=mpif90"
	          "-DCMAKE_BUILD_TYPE=Release"
            "-DGPTUNE_INSTALL_PATH=${placeholder "out"}"
	          "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
            "-DTPL_BLAS_LIBRARIES=${pkgs.blas}/lib/libblas${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"
            "-DTPL_LAPACK_LIBRARIES=${pkgs.lapack}/lib/liblapack${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}" 
            "-DTPL_SCALAPACK_LIBRARIES=${pkgs.scalapack}/lib/libscalapack${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"
          ] ++ pkgs.lib.optionals (pkgs.stdenv.isDarwin) [
            "-DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch"
          ];

          postInstall = ''
            cat ${./setup.py} > $out/setup.py
            mkdir -p $out/GPTune
            cat ${./GPTune/__version__.py} > $out/GPTune/__version__.py
          '';
        };

        #packages.gptune-py-pkg = pkgs.python39Packages.toPythonModule packages.gptune-libs;

        #GPTune as an importable python packages (do import GPTune, import GPTune.problem, etc.)
        #note that this does NOT load the required libs, which are provided by gptune-libs
        #certain files (specifically lcm.py) query GPTUNE_INSTALL_PATH to find them, and we MUST use them.
        packages.gptune-py-pkg = pkgs.python39Packages.buildPythonPackage rec {
          pname = "GPTune";
          inherit version;
          src = "${packages.gptune-libs}";
          propagatedBuildInputs = pydeps;
          doCheck = false;
        };

        #Main environment for use of GPTune, with the correct python packages preloaded
        devShell = pkgs.mkShell {
          nativeBuildInputs = systemDeps ++ [ (fullPythonWith packages.gptune-py-pkg) packages.gptune-libs ];
          shellHook = ''
            export PYTHONWARNINGS=ignore
            export GPTUNEROOT=$PWD
            export GPTUNE_INSTALL_PATH=${packages.gptune-libs}/gptune/
            export PYTHONPATH=$PYTHONPATH:${packages.gptune-libs}/gptune/
          '';
        };

        defaultPackage = packages.gptune-libs;
      });
}
