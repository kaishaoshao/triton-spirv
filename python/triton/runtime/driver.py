from ..backends import backends, DriverBase

from typing import Any, Callable, Generic, TypeVar


def _create_driver() -> DriverBase:
    # by environment variable TRITON_SPIRV_BACKEND, we can select the SPIR-V backend as the active driver.
    import os:
    if os.getenv("TRITON_SPIRV_BACKEND", "0") == "1":
        if "spirv" not in backends:
            raise RuntimeError("TRITON_SPIRV_BACKEND is set, but the spirv backend is not available.")
        return backends["spirv"].driver
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
    # SPIR-V backend is a special case, it's driver should not be used if another GPU backend is active.
    if len(active_drivers) >= 2 and backends["spirv"].driver.is_active():
        print("Both SPIR-V and GPU backends are active. Using SPIR-V backend.")
        active_drivers.remove(backends['spirv'].driver)
    if len(active_drivers) != 1:
        raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
    return active_drivers[0]()


T = TypeVar("T")


class LazyProxy(Generic[T]):

    def __init__(self, init_fn: Callable[[], T]) -> None:
        self._init_fn = init_fn
        self._obj: T | None = None

    def _initialize_obj(self) -> T:
        if self._obj is None:
            self._obj = self._init_fn()
        return self._obj

    def __getattr__(self, name) -> Any:
        return getattr(self._initialize_obj(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            setattr(self._initialize_obj(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._initialize_obj(), name)

    def __repr__(self) -> str:
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self) -> str:
        return str(self._initialize_obj())


class DriverConfig:

    def __init__(self) -> None:
        self.default: LazyProxy[DriverBase] = LazyProxy(_create_driver)
        self.active: LazyProxy[DriverBase] | DriverBase = self.default

    def set_active(self, driver: DriverBase) -> None:
        self.active = driver

    def reset_active(self) -> None:
        self.active = self.default


driver = DriverConfig()
