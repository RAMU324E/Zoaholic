"""配置 YAML 编解码工具。"""


# 迁移说明：
# 修改原因：该模块承载业务逻辑，不应继续放在 utils_pkg 这种通用工具包中。
# 修改方式：按照 Scout 的归位方案迁移到 core 对应业务模块，并只调整必要的内部导入路径。
# 目的：让业务代码按领域归属维护，同时保留根 utils.py 和 utils_pkg shim 的旧导入兼容性。
import yaml as _pyyaml


class _YamlHelper:
    """PyYAML CSafe wrapper — drop-in for the old ruamel YAML() instance."""
    def load(self, source):
        if hasattr(source, 'read'):
            return _pyyaml.load(source, Loader=_pyyaml.CSafeLoader)
        return _pyyaml.load(source, Loader=_pyyaml.CSafeLoader)

    def dump(self, data, stream):
        _pyyaml.dump(data, stream, Dumper=_pyyaml.CSafeDumper,
                     allow_unicode=True, default_flow_style=False,
                     sort_keys=False)


yaml = _YamlHelper()


def quote_colon_strings(obj):
    """
    递归处理配置数据（历史兼容 no-op）。
    PyYAML CSafeDumper 会自动给含冒号的字符串加引号，无需手动处理。
    """
    return obj


_quote_colon_strings = quote_colon_strings
