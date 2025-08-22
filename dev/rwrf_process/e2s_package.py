import logging

from package import PackageBundler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

variable = ["t2m", "u10", "qpepre"]  # HighRes vars
conditioning_variable = ["t2m"]  # LowRes vars
invariant = ["lsm", "orog"]  # HighRes invariants
y, x = 32, 32  # dimensions

package = PackageBundler(
    location="../e2s_package/",
    variable=variable,
    conditioning_variable=conditioning_variable,
    invariant=invariant,
    y=y,  # default set to 32
    x=x,  # default set to 32
)
package()
