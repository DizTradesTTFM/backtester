"""Contract specs for the futures we test.

point_value: dollars per 1.0 point move for one contract.
commission_rt: round-trip commission + exchange fees per contract (dollars).
These are typical retail/prop numbers. Adjust to match your firm.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    point_value: float
    commission_rt: float
    data_symbol: str = ""  # micros trade on the same prices as the full contract

    @property
    def data(self) -> str:
        return self.data_symbol or self.symbol

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


INSTRUMENTS = {
    "ES": Instrument("ES", 0.25, 50.0, 4.50),
    "NQ": Instrument("NQ", 0.25, 20.0, 4.50),
    "MES": Instrument("MES", 0.25, 5.0, 1.50, "ES"),
    "MNQ": Instrument("MNQ", 0.25, 2.0, 1.50, "NQ"),
}


def get_instrument(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError:
        raise ValueError(f"Unknown instrument {symbol!r}. Known: {sorted(INSTRUMENTS)}")
