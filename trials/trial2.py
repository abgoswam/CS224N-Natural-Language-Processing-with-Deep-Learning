class Item:
    def __init__(self, v):
        self.value = v


def show(items, name):
    print("---------Call : {0}----".format(name))
    for item in items:
        print(item.value)


if __name__ == '__main__':
    item1 = Item(1)
    item2 = Item(2)
    item3 = Item(3)

    items_a = [item1, item2, item3]
    items_b = items_a.copy()

    # display
    show(items_a, "a init")
    show(items_b, "b init")

    items_b[1].value = 100

    # display
    show(items_a, "a after change to b")
    show(items_b, "b after change to b")

    items_b.pop(1)

    # display
    show(items_a, "a after popping b")
    show(items_b, "b after popping b")

    del items_b[1]

    # display
    show(items_a, "a after del on b")
    show(items_b, "b after del on b")




