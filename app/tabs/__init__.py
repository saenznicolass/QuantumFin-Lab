from .yield_curve_tab import render_yield_curve_tab
from .trading_strategies_tab import render_trading_strategies_tab
from .portfolio_tab import render_portfolio_tab
from .option_pricing_tab import render_option_pricing_tab
from .model_comparison_tab import render_model_comparison_tab
# ...existing code...

__all__ = [
    "render_yield_curve_tab",
]
__all__.extend([
    "render_trading_strategies_tab",
    "render_portfolio_tab",
    "render_option_pricing_tab",
    "render_model_comparison_tab"
])