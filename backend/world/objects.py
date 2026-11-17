#TODO try to be samrt and make it easy to add more artibute and atribute types

from itertools import product

import numpy as np
import torch

from world.datatypes import Object


class ObjectSpace:
    def __init__(self, attributes: dict[str, list[str]]) -> None:
        self.attributes = attributes

    def __repr__(self) -> str:
        return f"ObjectSpace({list(self.attributes)})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ObjectSpace) and list(self.attributes.items()) == list(other.attributes.items())

    def __hash__(self) -> int:
        return hash(tuple((a, tuple(values)) for a, values in self.attributes.items()))

    def sizes(self) -> list[int]:
        return [len(values) for values in self.attributes.values()]

    def from_indices(self, idx) -> Object:
        idx = tuple(int(i) for i in idx)
        if len(idx) != len(self.attributes) or any(not 0 <= i < n for i, n in zip(idx, self.sizes())):
            raise ValueError(f"{idx} is not an object in {self}, sizes {self.sizes()}")
        return Object(idx, self)

    def make(self, **names: str) -> Object:
        if set(names) != set(self.attributes):
            raise ValueError(f"give exactly one value for each of {list(self.attributes)}, got {list(names)}")
        return self.from_indices(self.attributes[a].index(names[a]) for a in self.attributes)

    def all_objects(self) -> list[Object]:
        return [self.from_indices(idx) for idx in product(*(range(n) for n in self.sizes()))]

    def value(self, obj: Object, attribute: str) -> str:
        return self.attributes[attribute][obj.values[list(self.attributes).index(attribute)]]

    def name(self, obj: Object) -> str:
        return " ".join(self.value(obj, a) for a in self.attributes)

    def to_tensor(self, obj: Object) -> torch.Tensor:
        return torch.cat([torch.eye(n)[i] for i, n in zip(obj.values, self.sizes())])

    def split(self, held_out: float = 0.2, seed: int = 0) -> tuple[list[Object], list[Object]]:
        objects = self.all_objects()
        rng = np.random.default_rng(seed)
        n_held = int(held_out * len(objects))
        for _ in range(1000):
            order = rng.permutation(len(objects))
            held = [objects[i] for i in sorted(order[:n_held])]
            seen = [objects[i] for i in sorted(order[n_held:])]
            if [len({obj.values[a] for obj in seen}) for a in range(len(self.attributes))] == self.sizes():
                return seen, held
        raise ValueError("could not hold out that many objects and still show every value in the seen set")

    def score(self, obj: Object, guess: Object) -> float:
        return float(np.mean([a == b for a, b in zip(obj.values, guess.values, strict=True)]))
