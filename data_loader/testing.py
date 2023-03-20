import sys
import re
from random import shuffle


class data(object):

    def __init__(self, batch):
        self.batch = batch
        print(batch)
        self.pos = 0
        self.li = [x for x in range(10)]
        print(self.li)
        shuffle(self.li)


    # def __iter__(self):
    #     shuffle(self.li)
    #     return self
    def __len__(self):
        return 10

    def create_batch(self):
        x = [self.li[self.pos:self.pos+self.batch]]
        self.pos += self.batch
        return x
    def __next__(self):

        # self.pos += 1

        return self.create_batch()
def main():
    p = data(3)
    print(next(p))
    print(next(p))
    lis  = [x for x in range(10)]
    indec = [4,2,1]
    from operator import itemgetter
    h = list(itemgetter(*indec)(lis))
    print('ans',h)
    # print(next(p))
    # for i in p:
    #     print(i)

if __name__ == '__main__':
    main()