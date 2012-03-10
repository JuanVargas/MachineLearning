from cgen import *
from codepy.bpl import BoostPythonModule
mod = BoostPythonModule()

mod.add_function(
        FunctionBody(
            FunctionDeclaration(Const(Pointer(Value("char", "greet"))), []),
            Block([Statement('return "hello world"')])
            ))

from codepy.jit import guess_toolchain
cmod = mod.compile(guess_toolchain(), wait_on_error=True)

print cmod.greet()


