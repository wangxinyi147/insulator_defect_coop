# cloud/modules/__init__.py
"""
自定义模块 - 注册到 Ultralytics 命名空间
包含：注意力模块(LSKA, CA, CBAM, ECA)、轻量化卷积(GhostConv)、特征融合(ASFF)
"""
from .lska import LSKA
from .ca import CA
from .cbam import CBAM
from .eca import ECA
from .ghostconv import GhostConv
from .asff import ASFF


def register_all():
    """将自定义模块注入 Ultralytics 命名空间（modules + tasks globals）"""
    import ultralytics.nn.modules as unn
    import ultralytics.nn.tasks as tasks

    modules = {
        'LSKA': LSKA, 'CA': CA, 'CBAM': CBAM, 'ECA': ECA,
        'GhostConv': GhostConv, 'ASFF': ASFF,
    }

    # 1) 注入 ultralytics.nn.modules
    for name, cls in modules.items():
        setattr(unn, name, cls)
    if hasattr(unn, 'block'):
        for name in ('LSKA', 'CA', 'CBAM', 'ECA', 'ASFF'):
            setattr(unn.block, name, modules[name])
    if hasattr(unn, 'conv'):
        unn.conv.GhostConv = GhostConv

    # 2) 注入 ultralytics.nn.tasks 全局命名空间（parse_model 用 globals() 查类）
    for name, cls in modules.items():
        setattr(tasks, name, cls)

    print("[模块注册] LSKA, CA, CBAM, ECA, GhostConv, ASFF 已注册到 Ultralytics 命名空间")


__all__ = ["LSKA", "CA", "CBAM", "ECA", "GhostConv", "ASFF", "register_all"]
