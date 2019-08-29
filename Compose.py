# -*- coding: utf-8 -*-


class Compose():
    """Bundles several transforms.

    Args:
        transforms (list of transforms)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events):
        for t in self.transforms:
            events = t(events)
        return events

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
