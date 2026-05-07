"""
PlotProcesser class for seismic wave visualization.
PlotProcesser -> PlotterWrapperMixin -> MatplotlibPlotter/PlotlyPlotter
"""

from src.plotting.wrapper import PlotterWrapperMixin


class PlotProcesser(PlotterWrapperMixin):
    def __init__(self, **kwargs):

        # kwargs から 'backend' を取り出す（指定がなければ 'mpl' をデフォルトにする）
        target_backend = kwargs.pop("backend", "mpl")

        # Mixinのメソッドを呼び出してプロッタを生成
        # 残った kwargs (例: dpi=300 など) も一緒に渡してあげる
        self.init_plotter(backend=target_backend, **kwargs)
