function build()
%BUILD  Compile the aompls MEX file. Run from the matlab/ directory.

cd(fileparts(mfilename('fullpath')));
includes = ['-I' fullfile('..', 'cpp', 'include')];
% MATLAB R2018a+ supports C++17 via CXXFLAGS.
if ispc
    mex('-largeArrayDims', includes, 'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /DEIGEN_NO_DEBUG /DEIGEN_DONT_PARALLELIZE"', 'aompls_mex.cpp');
else
    mex('-largeArrayDims', includes, 'CXXFLAGS="$CXXFLAGS -std=c++17 -O3 -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE -Wno-maybe-uninitialized"', 'aompls_mex.cpp');
end
fprintf('Built aompls_mex (run test_parity() to validate).\n');
end
