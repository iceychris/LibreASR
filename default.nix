with import <nixpkgs> {};
stdenv.mkDerivation rec {
name = "env";
env = buildEnv { name = name; paths = buildInputs; };
buildInputs = [
gnumake

grpc

# go
go

# py
python37Full
python37Packages.black
python37Packages.pylint
python37Packages.setuptools
python37Packages.wheel
python37Packages.grpcio
python37Packages.grpcio-tools
python37Packages.tornado

# yt
youtube-dl
];
shellHook = ''
export GOPATH=/home/chris/go;
export PATH=$GOPATH/bin:$PATH;
'';
}

