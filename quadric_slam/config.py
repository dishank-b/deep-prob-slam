from typing import Any 

class Config(dict):
    """
    Config object with augmented 
    attribute access
    """
    def __init__(self, dictionary: dict = None, **kargs) -> None:
        super().__init__(dictionary, **kargs)

    def __getattribute__(self, __name: str) -> Any:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setattr__(__name, __value)

if __name__ == '__main__':
    config = {"a": 1}
    config1 = Config(config)
    config2 = Config({"a": 1})
    assert config1["a"] == config1.a
    assert config2["a"] == config2.a