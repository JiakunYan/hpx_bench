from spack import *
import os

class Hpx_bench(CMakePackage):
    """HPX Benchmarks"""

    homepage = "https://github.com/JiakunYan/lcw.git"
    git      = "https://github.com/JiakunYan/lcw.git"

    maintainers("JiakunYan")

    version('master', branch='master')

    def is_positive_int(val):
        try:
            return int(val) > 0
        except ValueError:
            return val == 'auto'

    variant('backend', default='mpi,lci', values=('mpi', 'lci'), multi=True,
            description='Communication backend')
    variant('shared', default=True,  description='Build with shared libraries')
    variant('examples', default=True, description='Build LCW examples')
    variant('cache-line', default='auto', values=is_positive_int,
            description='Cache line size, in bytes')
    variant('debug', default=False, description='Enable debug mode')

    depends_on("mpi", when="backend=mpi")
    depends_on("lci")

    def cmake_args(self):
        args = [
            self.define_from_variant('BUILD_SHARED_LIBS', 'shared'),
            self.define_from_variant('LCW_WITH_EXAMPLES', 'examples'),
            self.define_from_variant('LCW_DEBUG', 'debug'),
        ]

        if self.spec.satisfies("backend=mpi"):
            arg = self.define('LCW_TRY_ENABLE_BACKEND_MPI', True)
            args.append(arg)
        else:
            arg = self.define('LCW_TRY_ENABLE_BACKEND_MPI', False)
            args.append(arg)

        if self.spec.satisfies("backend=lci"):
            arg = self.define('LCW_TRY_ENABLE_BACKEND_LCI', True)
            args.append(arg)
        else:
            arg = self.define('LCW_TRY_ENABLE_BACKEND_LCI', False)
            args.append(arg)

        if self.spec.variants['cache-line'].value != 'auto':
            arg = self.define_from_variant('LCW_CACHE_LINE', 'cache-line')
            args.append(arg)

        return args
