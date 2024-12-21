from spack import *
import os

class HpxBench(CMakePackage):
    """HPX Benchmarks"""

    homepage = "https://github.com/JiakunYan/hpx_bench.git"
    git      = "https://github.com/JiakunYan/hpx_bench.git"

    maintainers("JiakunYan")

    version('master', branch='master')

    depends_on("hpx")
    depends_on("fftw")
